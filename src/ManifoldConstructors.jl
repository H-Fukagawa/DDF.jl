module ManifoldConstructors

using LinearAlgebra
using Random
using SparseArrays
using StaticArrays

using ..Manifolds
using ..Meshing
using ..SparseOps
using ..ZeroOrOne

################################################################################

export empty_manifold
"""
The empty manifold
"""
function empty_manifold(::Val{D}, ::Type{S}) where {D,S}
    return Manifold("empty manifold", zero(SparseOp{0,D,One}, 0, 0),
                    zeros(SVector{D,S}, 0))
end

################################################################################

export simplex_manifold
"""
Manifold with one standard simplex
"""
function simplex_manifold(::Val{D}, ::Type{S}) where {D,S}
    nvertices = D + 1
    coords = regular_simplex(Val(D), S)
    I = Int[]
    J = Int[]
    V = One[]
    for i in 1:nvertices
        push!(I, i)
        push!(J, 1)
        push!(V, One())
    end
    simplices = SparseOp{0,D,One}(sparse(I, J, V, nvertices, 1))
    return Manifold("simplex manifold", simplices, coords)
end

"""
Generate the coordinate positions for a regular D-simplex with edge
length 1.

The algorithm proceeds recursively. A 0-simplex is a point. A
D-simplex is a (D-1)-simplex that is shifted down along the new axis,
plus a new point on the new axis.
"""
function regular_simplex(::Val{D}, ::Type{S}) where {D,S}
    @assert D ≥ 0
    N = D + 1
    s = Array{SVector{D,S}}(undef, N)
    if D > 0
        s0 = regular_simplex(Val(D - 1), S)
        # Choose height so that edge length is 1
        if D == 1
            z = S(1)
        else
            z0 = sqrt(1 - sum(s0[1] .^ 2))
            if S <: Rational
                z = rationalize(typeof(zero(S).den), z0; tol=sqrt(eps(z0)))
            else
                z = z0::S
            end
        end
        z0 = -z / (D + 1)
        for n in 1:(N - 1)
            s[n] = SVector{D,S}(s0[n]..., z0)
        end
        s[N] = SVector{D,S}(zero(SVector{D - 1,S})..., z + z0)
    end
    return s
end

################################################################################

export orthogonal_simplex_manifold
"""
Manifold with one orthogonal simplex
"""
function orthogonal_simplex_manifold(::Val{D}, ::Type{S}) where {D,S}
    N = D + 1
    coords = orthogonal_simplex(Val(D), S)
    I = Int[]
    J = Int[]
    V = One[]
    for i in 1:N
        push!(I, i)
        push!(J, 1)
        push!(V, One())
    end
    simplices = SparseOp{0,D,One}(sparse(I, J, V, N, 1))
    return Manifold("orthogonal simplex manifold", simplices, coords)
end

"""
Generate the coordinate positions for an orthogonal D-simplex with
edge lengths 1.
"""
function orthogonal_simplex(::Val{D}, ::Type{S}) where {D,S}
    @assert D ≥ 0
    s = zeros(SVector{D,S}, D + 1)
    for d in 1:D
        s[d + 1] = setindex(zero(SVector{D,S}), 1, d)
    end
    return s
end

################################################################################

export hypercube_manifold
"""
Standard tesselation of a hypercube
"""
function hypercube_manifold(::Val{D}, ::Type{S}) where {D,S}
    @assert D ≥ 0
    N = D + 1

    # Find simplices
    simplices = SVector{N,Int}[]
    corner = zeros(SVector{D,Bool})
    vertices = [corner2vertex(corner)]
    next_corner!(simplices, vertices, corner)
    nsimplices = length(simplices)
    @assert nsimplices == factorial(D)

    # Set up coordinates
    coords = SVector{D,S}[]
    imin = CartesianIndex(ntuple(d -> 0, D))
    imax = CartesianIndex(ntuple(d -> 1, D))
    for i in imin:imax
        push!(coords, SVector{D,S}(i.I))
    end
    nvertices = length(coords)
    @assert nvertices == 2^D

    I = Int[]
    J = Int[]
    V = One[]
    for (j, sj) in enumerate(simplices)
        for i in sj
            push!(I, i)
            push!(J, j)
            push!(V, One())
        end
    end
    simplices = SparseOp{0,D,One}(sparse(I, J, V, nvertices, nsimplices))

    return Manifold("hypercube manifold", simplices, coords)
end

"""
- Accumulate the simplices ∈ `simplices`.
- `vertices` is the current set of vertices as we sweep from the
  origin to diagonally opposide vertex.
- `corner` is the current corner.
"""
function next_corner!(simplices::Vector{SVector{N,Int}}, vertices::Vector{Int},
                      corner::SVector{D,Bool})::Nothing where {N,D}
    @assert N == D + 1
    if D == 0
        @assert length(vertices) == 1
    end
    @assert count(corner) == length(vertices) - 1
    if length(vertices) == D + 1
        # We have all vertices; build the simplex
        push!(simplices, SVector{N}(vertices))
        return
    end
    # Loop over all neighbouring corners
    for d in 1:D
        if !corner[d]
            new_corner = setindex(corner, true, d)
            new_vertex = corner2vertex(new_corner)
            new_vertices = [vertices; new_vertex]
            next_corner!(simplices, new_vertices, new_corner)
        end
    end
end

function corner2vertex(c::SVector{D,Bool})::Int where {D}
    D == 0 && return 1
    return 1 + sum(c[d] << (d - 1) for d in 1:D)
end

################################################################################

export delaunay_hypercube_manifold
"""
Delaunay triangulation of a hypercube
"""
function delaunay_hypercube_manifold(::Val{D}, ::Type{S}) where {D,S}
    @assert D ≥ 0
    N = D + 1

    # Set up coordinates
    coords = SVector{D,S}[]
    imin = CartesianIndex(ntuple(d -> 0, D))
    imax = CartesianIndex(ntuple(d -> 1, D))
    for i in imin:imax
        # Re-map coordinates to avoid round-off errors
        push!(coords, SVector{D,S}(i.I) .+ S(5) / 2)
    end
    nvertices = length(coords)
    @assert nvertices == 2^D

    simplices = SparseOp{0,D,One}(delaunay_mesh(coords))

    return Manifold("delaunay hypercube manifold", simplices, coords)
end

################################################################################

export large_delaunay_hypercube_manifold
"""
Delaunay triangulation of a large hypercube
"""
function large_delaunay_hypercube_manifold(::Val{D}, ::Type{S}) where {D,S}
    @assert D ≥ 0
    N = D + 1

    ns = Dict{Int,Int}(0 => 1, 1 => 1024, 2 => 32, 3 => 16, 4 => 4, 5 => 2)
    n = ns[D]

    rng = MersenneTwister(1)

    # Set up coordinates
    coords = SVector{D,S}[]
    imin = CartesianIndex(ntuple(d -> 0, D))
    imax = CartesianIndex(ntuple(d -> n, D))
    for i in imin:imax
        # x = SVector{D,S}(i.I) / n
        dx = SVector{D,S}(i[d] == 0 || i[d] == n ? 0 :
                          S(rand(rng, -256:256)) / 65536 for d in 1:D)
        x = SVector{D,S}(i[d] + dx[d] for d in 1:D) / n
        push!(coords, x)
    end
    nvertices = length(coords)
    @assert nvertices == (n + 1)^D

    simplices = delaunay_mesh(coords)
    simplices = SparseOp{0,D,One}(simplices)

    return Manifold("large delaunay hypercube manifold", simplices, coords)
end

################################################################################

export refined_manifold
"""
Refined manifold
"""
function refined_manifold(mfd0::Manifold{D,C,S}) where {D,C,S}
    D == 0 && return mfd0
    coords = refine_coords(mfd0.lookup[(0, 1)], mfd0.coords[0])
    simplices = SparseOp{0,D}(delaunay_mesh(coords))
    mfd = Manifold("refined $(mfd0.name)", simplices, coords)
    return mfd
end

################################################################################

export refined_simplex_manifold
"""
Refined simplex manifold
"""
function refined_simplex_manifold(::Val{D}, ::Type{S}) where {D,S}
    mfd0 = simplex_manifold(Val(D), S)
    mfd = refined_manifold(mfd0)
    return mfd
end

################################################################################

export boundary_manifold
"""
Boundar manifold
"""
function boundary_manifold(mfd0::Manifold{D,C,S}) where {D,C,S}
    @assert D > 0
    # Find boundary faces
    B = mfd0.boundaries[D]::SparseOp{D - 1,D,Int8}
    volume_weights = ones(Int8, nsimplices(mfd0, D))
    keep_face = B.op * volume_weights
    keep_face::Vector{Int8}
    @assert all(x -> abs(x) ≤ 1, keep_face)
    faces_old2new = zeros(Int, nsimplices(mfd0, D - 1))
    faces_new2old = Int[]
    for j in 1:nsimplices(mfd0, D - 1)
        if keep_face[j] ≠ 0
            push!(faces_new2old, j)
            faces_old2new[j] = length(faces_new2old)
        end
    end
    nfaces_new = length(faces_new2old)
    # Find boundary vertices
    F = mfd0.simplices[D - 1]::SparseOp{0,D - 1,One}
    keep_vertex = falses(nsimplices(mfd0, 0))
    for j in faces_new2old
        for i in sparse_column_rows(F, j)
            keep_vertex[i] = true
        end
    end
    vertices_old2new = zeros(Int, nsimplices(mfd0, 0))
    vertices_new2old = Int[]
    for i in 1:nsimplices(mfd0, 0)
        if keep_vertex[i]
            push!(vertices_new2old, i)
            vertices_old2new[i] = length(vertices_new2old)
        end
    end
    nvertices_new = length(vertices_new2old)
    # Define new manifold
    name = "boundary $(mfd0.name)"
    I = Int[]
    J = Int[]
    V = One[]
    for jnew in 1:nfaces_new
        jold = faces_new2old[jnew]
        for iold in sparse_column_rows(F, jold)
            inew = vertices_old2new[iold]
            push!(I, inew)
            push!(J, jnew)
            push!(V, One())
        end
    end
    simplices = SparseOp{0,D - 1,One}(sparse(I, J, V, nvertices_new,
                                             nfaces_new))
    coords = mfd0.coords[0][vertices_new2old]
    mfd = Manifold(name, simplices, coords)
    return mfd
end

################################################################################

export boundary_simplex_manifold
"""
Boundary simplex manifold
"""
function boundary_simplex_manifold(::Val{D}, ::Type{S}) where {D,S}
    mfd0 = simplex_manifold(Val(D + 1), S)
    mfd = boundary_manifold(mfd0)
    return mfd
end

end
