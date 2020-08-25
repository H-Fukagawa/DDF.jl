using DDF

using LinearAlgebra

@testset "empty Manifolds D=$D" for D in 0:Dmax
    S = Rational{Int64}
    mfd = empty_manifold(Val(D), S)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == 0
    end
end

@testset "simplex Manifolds D=$D" for D in 0:Dmax
    # S = Rational{Int128}
    S = Float64
    mfd = simplex_manifold(Val(D), S)
    @test invariant(mfd)
    for R in 0:D
        @test nsimplices(mfd, R) == binomial(D + 1, R + 1)
    end

    N = D + 1
    for i in 1:N, j in (i + 1):N
        # @test isapprox(sum(((@view mfd.coords[i, :]) - (@view mfd.coords[j, :])) .^
        #                    2), 1; atol = sqrt(sqrt(eps())))
        @test norm((@view mfd.coords[i, :]) - (@view mfd.coords[j, :])) ≈ 1
    end

    # <https://en.wikipedia.org/wiki/Simplex#Volume>
    @test sum(mfd.volumes[D]) ≈ sqrt(S(D + 1) / 2^D) / factorial(D)
end

@testset "hypercube Manifolds D=$D" for D in 0:Dmax
    S = Rational{Int64}
    mfd = hypercube_manifold(Val(D), S)
    @test invariant(mfd)
    @test nsimplices(mfd, D) == factorial(D)
    @test nsimplices(mfd, 0) == 2^D
    # TODO: Find rule for other dimensions

    @test sum(mfd.volumes[D]) ≈ 1
end
