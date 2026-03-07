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
end
