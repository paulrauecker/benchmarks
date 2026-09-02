{
  description = "LLM benchmark harness (inspect_ai) for self-hosted and OpenRouter models";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python313
            pkgs.uv
            pkgs.jq
            pkgs.cacert
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];
          # Docker itself (daemon + CLI) is a host prerequisite for the code/agentic
          # sandbox evals, deliberately not provided here: pkgs.docker-client pulled
          # in docker-28.5.2, which this nixpkgs channel flags insecure and refuses
          # to build without --impure/NIXPKGS_ALLOW_INSECURE=1. The host's own
          # `docker` (confirmed present and working) is used instead.

          shellHook = ''
            # Nix devShells have no system CA bundle by default, so any dependency
            # using raw urllib (nltk's punkt downloader, pulled in by inspect_evals'
            # IFEval scorer) fails cert verification without this.
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            export NIX_SSL_CERT_FILE="$SSL_CERT_FILE"

            # numpy's (and other compiled deps') manylinux wheels dynamically
            # link libstdc++.so.6, which a pure Nix devShell doesn't put on
            # the loader path by default.
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            export UV_PYTHON_DOWNLOADS=never

            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              # Pin the base interpreter here only, via --python. Exporting
              # UV_PYTHON instead makes every later `uv pip install` target
              # that immutable nix-store interpreter directly (bypassing the
              # activated .venv below) and fail with "externally managed".
              uv venv --python "${pkgs.python313}/bin/python3.13"
            fi

            source .venv/bin/activate

            # -e so edits to src/ take effect without reinstalling
            if [ ! -f .venv/.installed ] || [ pyproject.toml -nt .venv/.installed ]; then
              echo "Installing dependencies..."
              uv pip install -e . && touch .venv/.installed
            fi

            [ -f .env ] && set -a && . ./.env && set +a

            echo "llm_bench ready. 'llm-bench --help' to start."
          '';
        };
      }
    );
}
