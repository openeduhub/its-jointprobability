{
  description = "A Bayesian approach to metadata prediction in education";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
    openapi-checks = {
      url = "github:openeduhub/nix-openapi-checks";
      inputs = {
        flake-utils.follows = "flake-utils";
      };
    };
    its-data = {
      url = "github:openeduhub/its-data";
      inputs = {
        flake-utils.follows = "flake-utils";
        nixpkgs.follows = "nixpkgs";
        nix-filter.follows = "nix-filter";
      };
    };
    model = {
      url = "gitlab:jopitz/its-jointprobability-model?host=gitlab.gwdg.de";
      inputs = {
        flake-utils.follows = "flake-utils";
        nixpkgs.follows = "nixpkgs";
        nix-filter.follows = "nix-filter";
      };
    };
  };

  nixConfig = {
    # additional caches for CUDA packages.
    # these packages are not included in the normal Nix cache, as they run
    # under an unfree license
    trusted-substituters = [
      "https://numtide.cachix.org"
      "https://cuda-maintainers.cachix.org"
    ];
    trusted-public-keys = [
      "numtide.cachix.org-1:2ps1kLBUWjxIneOy1Ik6cQjb41X0iXVXeHigGmycPPE="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    {
      # import overlays to add the service or its library to nixpkgs
      overlays = import ./overlays.nix {
        inherit (nixpkgs) lib;
        nix-filter = self.inputs.nix-filter.lib;
        its-data-overlay = self.inputs.its-data.overlays.default;
      };
    }
    // flake-utils.lib.eachDefaultSystem (
      system:
      let
        get-pkgs =
          cudaSupport:
          import nixpkgs {
            inherit system;
            config = {
              inherit cudaSupport;
              allowUnfree = true;
            };
            overlays = [
              (final: prev: {
                pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
                  (py-final: py-prev: {
                    # manually provide py3langid here, because we are using an
                    # old version of nixpkgs
                    py3langid = py-final.callPackage ./pkgs/py3langid.nix { };
                  })
                ];
              })
              self.outputs.overlays.python-lib
              self.outputs.overlays.its-jointprobability
              self.inputs.model.overlays.default
            ];
          };

        pkgs-with-cuda = get-pkgs true;
        pkgs-without-cuda = get-pkgs false;
      in
      {
        # packages that we can build
        packages =
          rec {
            default = its-jointprobability;
            inherit (pkgs-without-cuda) its-jointprobability;
            optuna-env = pkgs-without-cuda.python3Packages.its-jointprobability-with-oputuna;
          }
          //
          # CUDA is only supported on x86_64 linux
          (nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
            with-cuda = pkgs-with-cuda.its-jointprobability;
            optuna-env-with-cuda = pkgs-with-cuda.python3Packages.its-jointprobability-with-oputuna;
          });

        # additional binaries we can run
        apps =
          {
            retrain-model = {
              type = "app";
              program = "${pkgs-without-cuda.its-jointprobability}/bin/retrain-model";
            };
            run-study = {
              type = "app";
              program = "${pkgs-without-cuda.its-jointprobability-with-optuna}/bin/study-prodslda";
            };
          }
          //
          # CUDA is only supported on x86_64 linux
          (nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
            retrain-model-with-cuda = {
              type = "app";
              program = "${pkgs-with-cuda.its-jointprobability}/bin/retrain-model";
            };
            run-study-with-cuda = {
              type = "app";
              program = "${pkgs-with-cuda.its-jointprobability-with-optuna}/bin/study-prodslda";
            };
          });

        # the development environment
        devShells =
          {
            default = pkgs-without-cuda.callPackage ./shell.nix { nix-filter = self.inputs.nix-filter.lib; };
          }
          //
          # CUDA is only supported on x86_64 linux
          (nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
            with-cuda = pkgs-with-cuda.callPackage ./shell.nix { nix-filter = self.inputs.nix-filter.lib; };
          });

        # integration testing
        checks =
          { }
          // (nixpkgs.lib.optionalAttrs
            # only run the VM checks on linux systems
            (system == "x86_64-linux" || system == "aarch64-linux")
            {
              openapi-check =
                let
                  openapi-checks = self.inputs.openapi-checks.lib.${system};
                in
                (openapi-checks.test-service {
                  service-bin = "${pkgs-without-cuda.its-jointprobability} ${pkgs-without-cuda.its-jointprobability-model} --debug";
                  service-port = 8080;
                  openapi-domain = "/openapi.json";
                  memory-size = 4 * 1024;
                });
            }
          );
      }
    );
}
