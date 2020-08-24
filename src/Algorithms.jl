module Algorithms

using DifferentialForms
using StaticArrays

# function circumcentre1(xs::SVector{R, <:Chain{V, 1, T}}) where {R, V, T}
#     # G. Westendorp, A formula for the N-circumsphere of an N-simplex,
#     # <https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm>,
#     # April 2013.
#     @assert iseuclidean(V)
#     D = ndims(V)
#     @assert R == D + 1
# 
#     # Convert Euclidean to conformal basis
#     cxs = conformal.(xs)
#     # Circumsphere (this formula is why we are using conformal GA)
#     X = ∧(cxs)
#     # Hodge dual
#     sX = ⋆X
#     # Euclidean part is centre
#     cc = euclidean(sX)
# 
#     # Calculate radius
#     # TODO: Move this into a test case
#     r2 = scalar(abs2(cc)).v - 2 * sX.v[1]
#     # Check radii
#     for i in 1:R
#         ri2 = scalar(abs2(xs[i] - cc)).v
#         @assert abs(ri2 - r2) <= T(1.0e-12) * r2
#     end
# 
#     cc::Chain{V, 1, T}
# end

export circumcentre
function circumcentre(xs::SVector{N,<:Form{D,1,T}}) where {N,D,T}
    # See arXiv:1103.3076v2 [cs.RA], section 10.1
    A = SMatrix{N + 1,N + 1}(i <= N && j <= N ? 2 * (xs[i] ⋅ xs[j])[] :
                             i == j ? zero(T) : one(T)
                             for i in 1:(N + 1), j in 1:(N + 1))
    b = SVector{N + 1}(i <= N ? (xs[i] ⋅ xs[i])[] : one(T) for i in 1:(N + 1))
    c = A \ b
    cc = sum(c[i] * xs[i] for i in 1:N)
    return cc::Form{D,1,T}
end

end
