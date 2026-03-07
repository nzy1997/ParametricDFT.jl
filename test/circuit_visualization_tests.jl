using CairoMakie

@testset "Circuit Visualization" begin
    @testset "plot_circuit QFTBasis" begin
        basis = QFTBasis(2, 2)
        fig = plot_circuit(basis)
        @test fig isa CairoMakie.Figure

        # With output_path
        path = tempname() * ".png"
        try
            fig2 = plot_circuit(basis; output_path=path)
            @test fig2 isa CairoMakie.Figure
            @test isfile(path)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "plot_circuit EntangledQFTBasis" begin
        basis = EntangledQFTBasis(2, 2; entangle_position=:back)
        fig = plot_circuit(basis)
        @test fig isa CairoMakie.Figure

        # Test different entangle positions
        for pos in (:front, :middle, :back)
            b = EntangledQFTBasis(2, 2; entangle_position=pos)
            f = plot_circuit(b)
            @test f isa CairoMakie.Figure
        end
    end

    @testset "plot_circuit TEBDBasis" begin
        basis = TEBDBasis(2, 2)
        fig = plot_circuit(basis)
        @test fig isa CairoMakie.Figure

        # With output_path
        path = tempname() * ".png"
        try
            fig2 = plot_circuit(basis; output_path=path)
            @test fig2 isa CairoMakie.Figure
            @test isfile(path)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "plot_circuit MERABasis" begin
        basis = MERABasis(2, 2)
        fig = plot_circuit(basis)
        @test fig isa CairoMakie.Figure

        # With output_path
        path = tempname() * ".png"
        try
            fig2 = plot_circuit(basis; output_path=path)
            @test fig2 isa CairoMakie.Figure
            @test isfile(path)
        finally
            isfile(path) && rm(path)
        end
    end
end
