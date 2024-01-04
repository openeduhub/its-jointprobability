{
  description = "A Bayesian approach to metadata prediction in education";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
    nlprep = {
      url = "github:openeduhub/nlprep";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    openapi-checks = {
      url = "github:openeduhub/nix-openapi-checks";
      inputs = {
        flake-utils.follows = "flake-utils";
      };
    };
  };

  nixConfig = {
    trusted-substituters = [
      "https://numtide.cachix.org"
      "https://cuda-maintainers.cachix.org"
    ];
    trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "numtide.cachix.org-1:2ps1kLBUWjxIneOy1Ik6cQjb41X0iXVXeHigGmycPPE="
    ];
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    {
      # define an overlay to add text-extraction to nixpkgs
      overlays.default = (final: prev: {
        its-jointprobability = self.packages.${final.system}.webservice;
      });
    } //
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs-with-cuda = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        pkgs-without-cuda = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = false;
          };
        };
        openapi-checks = self.inputs.openapi-checks.lib.${system};
        nix-filter = self.inputs.nix-filter.lib;
        get-python = pkgs: pkgs.python310;

        ### list of python packages required to build / run the application
        python-packages-build = py-pkgs:
          with py-pkgs; [
            setuptools
            pandas
            uvicorn
            pydantic
            fastapi
            pyro-ppl
            icecream
            matplotlib
            optuna
            (self.inputs.nlprep.lib.${system}.nlprep py-pkgs) # nlp pre-processing
          ];

        ### list of python packages to include in the development environment
        # the development installation contains all build packages,
        # plus some additional ones we do not need to include in production.
        python-packages-devel = py-pkgs:
          with py-pkgs; [
            black
            pyflakes
            isort
            pylint
            ipython
            mypy
            # library stubs for mypy
            types-tqdm
            pandas-stubs
          ]
          ++ (python-packages-build py-pkgs);

        ### create the python package
        python-pkg-lib = py-pkgs: py-pkgs.buildPythonPackage {
          pname = "its-jointprobability";
          version = "0.1.1";
          /*
          only include files that are related to the application
          this will prevent unnecessary rebuilds
          */
          src = nix-filter {
            root = self;
            include = [
              # folders
              "its_jointprobability"
              # files
              ./setup.py
              ./requirements.txt
            ];
            exclude = [ (nix-filter.matchExt "pyc") ];
          };
          prePatch = ''
            substituteInPlace its_jointprobability/*.py \
              --replace "Path.cwd() / \"data\"" "Path(\"${./data}\")"
          '';
          propagatedBuildInputs = (python-packages-build py-pkgs);
          doCheck = false;
        };

        get-python-package = pkgs: python-pkg-lib (get-python pkgs).pkgs;
        get-python-application = pkgs: (get-python pkgs).pkgs.toPythonApplication (get-python-package pkgs);

        get-devShell = pkgs: pkgs.mkShell {
          buildInputs = [
            # the development installation of python
            ((get-python pkgs).withPackages python-packages-devel)
            # python LSP server
            pkgs.nodePackages.pyright
            # for automatically generating nix expressions, e.g. from PyPi
            pkgs.nix-template
            pkgs.nix-init
          ];
        };

      in
      {
        # the packages that we can build
        packages = rec {
          webservice = get-python-application pkgs-without-cuda;
          default = webservice;
        } //
        # CUDA is only supported on x86_64 linux
        (
          nixpkgs.lib.optionalAttrs
            (system == "x86_64-linux")
            rec {
              webservice-with-cuda = get-python-application pkgs-with-cuda;
              with-cuda = webservice-with-cuda;
            }
        );
        # the library that may be used in different python versions
        lib = {
          its-jointprobability = python-pkg-lib;
        };
        # the development environment
        devShells = {
          default = (get-devShell pkgs-without-cuda);
        } //
        # CUDA is only supported on x86_64 linux
        (
          nixpkgs.lib.optionalAttrs
            (system == "x86_64-linux")
            {
              with-cuda = (get-devShell pkgs-with-cuda);
            }
        );
        checks = { } // (nixpkgs.lib.optionalAttrs
          # only run the VM checks on linux systems
          (system == "x86_64-linux" || system == "aarch64-linux")
          {
            openapi-check = (
              openapi-checks.test-service {
                service-bin =
                  "${self.packages.${system}.webservice}/bin/its-jointprobability --debug";
                service-port = 8080;
                openapi-domain = "/openapi.json";
                memory-size = 2048;
              }
            );
          });
      }
    );
}
