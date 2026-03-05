# ============================================================================
# Tests for Basis Serialization (serialization.jl)
# ============================================================================

@testset "Basis Serialization" begin

    # Create temp directory for test files
    test_dir = mktempdir()

    # ========================================================================
    # Config-driven tests: common across all basis types
    # ========================================================================

    BASIS_CONFIGS = [
        (
            type = QFTBasis,
            make = () -> QFTBasis(4, 4),
            make_small = () -> QFTBasis(3, 3),
            version = "1.0",
            check_fields = (basis, loaded) -> nothing,
            check_dict = (d, basis) -> nothing,
            check_json = (json_data) -> nothing,
            sizes = [(2, 2), (3, 4), (5, 3), (4, 4)],
            check_size_fields = (loaded, m, n) -> nothing,
        ),
        (
            type = EntangledQFTBasis,
            make = () -> EntangledQFTBasis(4, 4; entangle_phases=[0.1, 0.2, 0.3, 0.4]),
            make_small = () -> EntangledQFTBasis(2, 2; entangle_phases=[0.5, 1.0]),
            version = "1.1",
            check_fields = (basis, loaded) -> begin
                @test loaded.n_entangle == basis.n_entangle
                @test loaded.entangle_phases ≈ basis.entangle_phases
            end,
            check_dict = (d, basis) -> begin
                @test d["n_entangle"] == basis.n_entangle
                @test d["entangle_phases"] ≈ basis.entangle_phases
                @test d["entangle_position"] == "back"
            end,
            check_json = (json_data) -> begin
                @test json_data.n_entangle == 2
                @test collect(json_data.entangle_phases) ≈ [0.5, 1.0]
                @test json_data.entangle_position == "back"
            end,
            sizes = [(2, 2), (3, 4), (4, 3)],
            check_size_fields = (loaded, m, n) -> begin
                @test loaded.n_entangle == min(m, n)
            end,
        ),
        (
            type = TEBDBasis,
            make = () -> TEBDBasis(3, 3; phases=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            make_small = () -> TEBDBasis(2, 2; phases=[0.1, 0.2, 0.3, 0.4]),
            version = "2.0",
            check_fields = (basis, loaded) -> begin
                @test loaded.n_row_gates == basis.n_row_gates
                @test loaded.n_col_gates == basis.n_col_gates
                @test loaded.phases ≈ basis.phases
            end,
            check_dict = (d, basis) -> begin
                @test d["n_row_gates"] == basis.n_row_gates
                @test d["n_col_gates"] == basis.n_col_gates
                @test d["phases"] ≈ basis.phases
            end,
            check_json = (json_data) -> begin
                @test json_data.n_row_gates == 2
                @test json_data.n_col_gates == 2
                @test collect(Float64, json_data.phases) ≈ [0.1, 0.2, 0.3, 0.4]
            end,
            sizes = [(2, 2), (3, 4), (4, 3)],
            check_size_fields = (loaded, m, n) -> begin
                @test loaded.n_row_gates == m
                @test loaded.n_col_gates == n
            end,
        ),
    ]

    for cfg in BASIS_CONFIGS
        type_name = string(cfg.type)

        @testset "$type_name save and load" begin
            basis = cfg.make()

            path = joinpath(test_dir, "test_$(type_name)_save.json")
            returned_path = save_basis(path, basis)
            @test returned_path == path
            @test isfile(path)

            loaded = load_basis(path)
            @test loaded isa cfg.type
            @test loaded.m == basis.m
            @test loaded.n == basis.n
            @test length(loaded.tensors) == length(basis.tensors)
            cfg.check_fields(basis, loaded)

            for (t1, t2) in zip(basis.tensors, loaded.tensors)
                @test t1 ≈ t2
            end

            @test basis_hash(loaded) == basis_hash(basis)
        end

        @testset "$type_name JSON structure" begin
            basis = cfg.make_small()
            path = joinpath(test_dir, "test_$(type_name)_json.json")
            save_basis(path, basis)

            json_str = read(path, String)
            json_data = JSON3.read(json_str)
            @test json_data.type == type_name
            @test json_data.version == cfg.version
            @test json_data.m == basis.m
            @test json_data.n == basis.n
            @test haskey(json_data, :tensors)
            @test haskey(json_data, :hash)
            cfg.check_json(json_data)
        end

        @testset "$type_name round-trip preserves functionality" begin
            Random.seed!(42)
            basis = cfg.make()

            path = joinpath(test_dir, "test_$(type_name)_roundtrip.json")
            save_basis(path, basis)
            loaded = load_basis(path)

            img_size = image_size(basis)
            img = rand(img_size...)
            freq_original = forward_transform(basis, img)
            freq_loaded = forward_transform(loaded, img)
            @test freq_original ≈ freq_loaded

            recovered_original = inverse_transform(basis, freq_original)
            recovered_loaded = inverse_transform(loaded, freq_loaded)
            @test recovered_original ≈ recovered_loaded
        end

        @testset "$type_name basis_to_dict and dict_to_basis" begin
            basis = cfg.make()

            d = basis_to_dict(basis)
            @test d isa Dict
            @test d["type"] == type_name
            @test d["version"] == cfg.version
            @test d["m"] == basis.m
            @test d["n"] == basis.n
            @test haskey(d, "tensors")
            @test haskey(d, "hash")
            cfg.check_dict(d, basis)

            loaded = dict_to_basis(d)
            @test loaded isa cfg.type
            @test loaded.m == basis.m
            @test loaded.n == basis.n
            @test basis_hash(loaded) == basis_hash(basis)
        end

        @testset "$type_name different sizes" begin
            for (m, n) in cfg.sizes
                basis = if cfg.type == EntangledQFTBasis
                    EntangledQFTBasis(m, n; entangle_phases=rand(min(m, n)))
                elseif cfg.type == TEBDBasis
                    TEBDBasis(m, n; phases=rand(m + n))
                else
                    QFTBasis(m, n)
                end
                path = joinpath(test_dir, "test_$(type_name)_size_$(m)_$(n).json")

                save_basis(path, basis)
                loaded = load_basis(path)

                @test loaded.m == m
                @test loaded.n == n
                @test image_size(loaded) == (2^m, 2^n)
                cfg.check_size_fields(loaded, m, n)
            end
        end
    end

    # ========================================================================
    # QFTBasis-specific tests
    # ========================================================================

    @testset "save trained basis" begin
        Random.seed!(42)

        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:3]
        trained_basis, _ = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=2
        )

        path = joinpath(test_dir, "trained_basis.json")
        save_basis(path, trained_basis)
        loaded = load_basis(path)

        @test loaded isa QFTBasis
        @test basis_hash(loaded) == basis_hash(trained_basis)

        img = rand(8, 8)
        freq_trained = forward_transform(trained_basis, img)
        freq_loaded = forward_transform(loaded, img)
        @test freq_trained ≈ freq_loaded
    end

    @testset "hash verification on load" begin
        basis = QFTBasis(3, 3)
        path = joinpath(test_dir, "test_hash.json")
        save_basis(path, basis)

        loaded = load_basis(path)
        @test basis_hash(loaded) == basis_hash(basis)
    end

    # ========================================================================
    # EntangledQFTBasis-specific tests (entangle_position)
    # ========================================================================

    @testset "EntangledQFTBasis entangle_position serialization" begin
        for pos in [:front, :middle, :back]
            phases = [0.1, 0.2, 0.3]
            basis = EntangledQFTBasis(3, 3; entangle_phases=phases, entangle_position=pos)

            path = joinpath(test_dir, "test_entangled_pos_$(pos).json")
            save_basis(path, basis)
            loaded_basis = load_basis(path)

            @test loaded_basis isa EntangledQFTBasis
            @test loaded_basis.entangle_position == pos
            @test loaded_basis.entangle_phases ≈ basis.entangle_phases
            @test basis_hash(loaded_basis) == basis_hash(basis)

            img = rand(8, 8)
            freq_original = forward_transform(basis, img)
            freq_loaded = forward_transform(loaded_basis, img)
            @test freq_original ≈ freq_loaded
        end
    end

    @testset "EntangledQFTBasis entangle_position in JSON" begin
        basis = EntangledQFTBasis(3, 3; entangle_position=:front)
        path = joinpath(test_dir, "test_entangled_pos_json.json")
        save_basis(path, basis)

        json_str = read(path, String)
        json_data = JSON3.read(json_str)
        @test json_data.entangle_position == "front"
    end

    @testset "EntangledQFTBasis entangle_position dict roundtrip" begin
        for pos in [:front, :middle, :back]
            basis = EntangledQFTBasis(3, 3; entangle_position=pos)
            d = basis_to_dict(basis)
            @test d["entangle_position"] == string(pos)

            loaded = dict_to_basis(d)
            @test loaded.entangle_position == pos
            @test basis_hash(loaded) == basis_hash(basis)
        end
    end

    @testset "EntangledQFTBasis backward compat (missing entangle_position)" begin
        basis = EntangledQFTBasis(3, 3)
        d = basis_to_dict(basis)
        delete!(d, "entangle_position")

        loaded = dict_to_basis(d)
        @test loaded isa EntangledQFTBasis
        @test loaded.entangle_position == :back
    end

    # Cleanup
    rm(test_dir, recursive=true)
end
