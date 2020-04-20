using ComputedFieldTypes
using SparseArrays



bitsign(b::Bool) = b ? -1 : 1
bitsign(i::Integer) = bitsign(isodd(i))



function sort_perm(xs::Vector{T})::Tuple{Vector{T}, Bool} where {T}
    n = length(xs)
    ys = copy(xs)
    s = false
    for i in 1:n
        imin = argmin(@view ys[i:n]) + i - 1
        t = ys[i]
        ys[i] = ys[imin]
        ys[imin] = t
        s = xor(s, isodd(i - imin))
    end
    @assert issorted(ys)
    ys, s
end

function sort_perm(xs::NTuple{D, T})::Tuple{NTuple{D, T}, Bool} where {D, T}
    # TODO: Make this efficient
    ys, s = sort_perm(collect(xs))
    tuple(ys...), s
end



export Simplex
@computed struct Simplex{D}
    vertices::NTuple{D+1, Int}
    signbit::Bool

    function Simplex{D}(vertices::NTuple{D1, Int},
                        signbit::Bool=false) where {D, D1}
        @assert D1 == D+1
        v, s = sort_perm(vertices)
        new{D}(v, xor(signbit, s))
    end
    function Simplex(vertices::NTuple{D1, Int}, signbit::Bool=false) where {D1}
        D = D1-1
        Simplex{D}(vertices, signbit)
    end
end

export invariant
function invariant(s::Simplex)::Bool
    issorted(s.vertices)
end

function Base.show(io::IO, s::Simplex)
    print(io, "($(s.vertices); $(bitsign(s.signbit)))")
end

Base.:(==)(s::S, t::S) where {S<:Simplex} =
    s.vertices == t.vertices && s.signbit == t.signbit
function Base.isless(s::S, t::S) where {S<:Simplex}
    isless(s.vertices, t.vertices) && return true
    isless(t.vertices, s.vertices) && return false
    isless(s.signbit, t.signbit)
end

export dim
dim(::Type{S}) where {S<:Simplex} = length(S) - 1
dim(::S) where {S<:Simplex} = dim(S)

Base.getindex(s::Simplex, i) = s.vertices[i]
Base.length(::Type{<:Simplex{D}}) where {D} = D + 1
Base.length(::S) where {S<:Simplex} = length(S)



function simplices_type(D::Int)::Type
    Tuple{(Vector{fulltype(Simplex{d})} for d in 1:D)...}
end

export Manifold
# """
# Manifold (aka Chain)
# """
@computed struct Manifold{D}
    nvertices::Int
    # vertices (i.e. 0-simplices) are always numbered 1:nvertices and
    # are not stored
    simplices::simplices_type(D)
    # The coderivative δ of 0-forms vanishes and is not stored
    coderivs::NTuple{D, SparseMatrixCSC{Int8, Int}}

    function Manifold{D}(nvertices::Int,
                         simplices::NTuple{D, Vector{<:Simplex}},
                         coderivs::NTuple{D, SparseMatrixCSC{Int8, Int}}
                         ) where {D}
        simplices::simplices_type(D)
        mf = new{D}(nvertices, simplices, coderivs)
        @assert invariant(mf)
        mf
    end
    function Manifold(nvertices::Int,
                      simplices::NTuple{D, Vector{<:Simplex}},
                      coderivs::NTuple{D, SparseMatrixCSC{Int8, Int}}) where {D}
        Manifold{D}(nvertices, simplices, coderivs)
    end
end
# TODO: Implement also the "cube complex" representation

export invariant
function invariant(mf::Manifold{D})::Bool where {D}
    D >= 0 || (@assert false; return false)

    mf.nvertices >= 0 || (@assert false; return false)

    for R in 1:D
        simplices = mf.simplices[R]
        for i in 1:length(simplices)
            s = simplices[i]
            for d in 1:R+1
                1 <= s[d] <= mf.nvertices || (@assert false; return false)
            end
            for d in 2:R+1
                s[d] > s[d-1] || (@assert false; return false)
            end
            if i > 1
                s > simplices[i-1] || (@assert false; return false)
            end
        end
    end

    for R in 1:D
        coderivs = mf.coderivs[R]
        size(coderivs) == (dim(R-1, mf), dim(R, mf)) ||
            (@assert false; return false)
    end

    return true
end

# Comparison

function Base.:(==)(mf1::Manifold{D}, mf2::Manifold{D})::Bool where {D}
    mf1.nvertices == mf2.nvertices || return false
    mf1.simplices == mf2.simplices
end

Base.ndims(::Manifold{D}) where {D} = D

function dim(::Val{R}, mf::Manifold{D})::Int where {R, D}
    R == 0 && return mf.nvertices
    length(mf.simplices[R])
end
function dim(R, mf::Manifold)::Int
    R == 0 && return mf.nvertices
    length(mf.simplices[R])
end

# Convenience constructors

function Manifold(simplices::Vector{Simplex{D, X}})::Manifold{D} where {D, X}
    # # Ensure simplex vertices are sorted
    # for s in simplices
    #     for a in 2:D+1
    #         @assert s[a] > s[a-1]
    #     end
    # end
    # # Ensure simplices are sorted
    # for i in 2:length(simplices)
    #     @assert simplices[i] > simplices[i-1]
    # end
    # Count vertices
    nvertices = 0
    for s in simplices
        for a in 1:D+1
            nvertices = max(nvertices, s[a])
        end
    end
    # # Ensure all vertices are mentioned (we could omit this check)
    # vertices = falses(nvertices)
    # for s in simplices
    #     for a in 1:D+1
    #         vertices[s[s]] = true
    #     end
    # end
    # @assert all(vertices)

    simplices = copy(simplices)
    sort!(simplices)
    unique!(simplices)
    if D == 0
        return Manifold{D}(nvertices, (), ())
    end

    # Calculate lower-dimensional simplices
    # See arXiv:1103.3076, section 7
    faces = fulltype(Simplex{D-1})[]
    coderivs1 = Tuple{fulltype(Simplex{D-1}), Int}[]
    for (i,s) in enumerate(simplices)
        for a in 1:D+1
            # Leave out vertex a
            v1 = ntuple(b -> s[b + (b>=a)], D)
            s1 = xor(s.signbit, isodd(a-1))
            face = Simplex{D-1}(v1)
            # face = Simplex{D-1}(face.vertices, false)
            coderiv1 = (Simplex{D-1}(v1, s1), i)
            push!(faces, face)
            push!(coderivs1, coderiv1)
        end
    end
    sort!(faces)
    unique!(faces)
    mf1 = Manifold(faces)

    sort!(coderivs1)
    @assert allunique(coderivs1)
    I = Int[]
    J = Int[]
    V = Int8[]
    i = 0
    lastv = nothing
    for (s,j) in coderivs1
        if s.vertices != lastv
            i += 1
            lastv = s.vertices
        end
        push!(I, i)
        push!(J, j)
        push!(V, bitsign(s.signbit))
    end
    @assert i == length(faces)
    coderivs = sparse(I, J, V)

    Manifold{D}(nvertices,
                tuple(mf1.simplices..., simplices),
                tuple(mf1.coderivs..., coderivs))
end

function Manifold(simplices::Vector{NTuple{D1, Int}})::Manifold{D1-1} where {D1}
    D = D1-1
    Manifold(fulltype(Simplex{D})[Simplex{D}(s) for s in simplices])
end

function Manifold(::Val{D})::Manifold{D} where {D}
    Manifold(fulltype(Simplex{D})[])
end

function Manifold(simplex::Simplex{D})::Manifold{D} where {D}
    Manifold([simplex])
end

# Boundary and derivative

export boundary
function boundary()
end

export deriv
function deriv()
end
