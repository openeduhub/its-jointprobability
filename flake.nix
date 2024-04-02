{
  description = "A Bayesian approach to metadata prediction in education";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
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

  outputs = { self, nixpkgs, flake-utils, ... }:
  {
    # define an overlay to add the service to nixpkgs
    overlays.default = (final: prev: {
      its-jointprobability = self.packages.${final.system}.webservice;
    });
  } //
  flake-utils.lib.eachDefaultSystem (system:
  let
    # import nixpkgs, optionally with CUDA support
    pkgs-with-cuda = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
      overlays = [
        self.inputs.its-data.overlays.default
      ];
    };
    pkgs-without-cuda = nixpkgs.legacyPackages.${system}.extend
    self.inputs.its-data.overlays.default;

    # some utility libraries
    openapi-checks = self.inputs.openapi-checks.lib.${system};
    nix-filter = self.inputs.nix-filter.lib;

    # use the default python version (3.11 for nixos 23.11 and 24.05)
    get-python = pkgs: pkgs.python3;

    # lists of python packages required to run the application, build the
    # model, or for development
    python-packages = import ./python-packages.nix;

    model = self.inputs.model.packages.${system}.its-jointprobability-model;

    get-src = excludeOptuna: nix-filter {
      root = self;
      # only include files that are related to the application.
      # this will prevent unnecessary rebuilds
      include = [
        (nix-filter.inDirectory ./its_jointprobability)
        ./setup.py
        ./requirements.txt
      ];
      exclude = [
        (nix-filter.matchExt "pyc")
      ]
      # optionally ignore hyperparameter-optimization related code
      ++
      (nixpkgs.lib.lists.optionals excludeOptuna [
        (nix-filter.inDirectory ./its_jointprobability/optuna)
      ]);
    };

    # the python package containing the deployable service
    get-python-lib-deploy = py-pkgs: py-pkgs.buildPythonPackage {
      pname = "its-jointprobability";
      version = "0.2.0";
      src = get-src true;
      # replace local lookups of the model with the model that we pulled in
      # the inputs
      prePatch = ''
        substituteInPlace its_jointprobability/*.py \
          --replace "Path.cwd() / \"data\"" \
                    "Path(\"${model}\")"
      '';
      propagatedBuildInputs = (python-packages.deploy-pkgs py-pkgs);
    };

    # the python package also containing the dependencies for hyperparameter
    # optimization
    get-python-lib-optuna = pkgs: py-pkgs:
      (get-python-lib-deploy py-pkgs).overrideAttrs (oldAttrs: {
        # no longer ignore the optuna module
        src = get-src false;
        # override the dependencies to include optuna-related packages and
        # sqlite
        propagatedBuildInputs = [
          pkgs.sqlite
        ] ++ (python-packages.optuna-pkgs py-pkgs);
      });

    # some simple wrappers for more easily creating the python applications
    get-python-app-deploy = pkgs:
      let py-pkgs = (get-python pkgs).pkgs;
      in
        py-pkgs.toPythonApplication (
          get-python-lib-deploy py-pkgs
      );
    get-python-app-optuna = pkgs:
      let py-pkgs = (get-python pkgs).pkgs;
      in
        py-pkgs.toPythonApplication (
          get-python-lib-optuna pkgs py-pkgs
      );

    # define the development environment
    get-devShell = pkgs: pkgs.mkShell {
      buildInputs = [
        # the development installation of python
        ((get-python pkgs).withPackages python-packages.devel-pkgs)
        # python LSP server
        pkgs.nodePackages.pyright
        # for automatically generating nix expressions, e.g. from PyPi
        pkgs.nix-template
        pkgs.nix-init
      ];
    };

  in
  {
    # packages that we can build
    packages = rec {
      webservice = get-python-app-deploy pkgs-without-cuda;
      optuna-env = get-python-app-optuna pkgs-without-cuda;
      default = webservice;
    } //
    # CUDA is only supported on x86_64 linux
    (
      nixpkgs.lib.optionalAttrs
      (system == "x86_64-linux")
      rec {
        webservice-with-cuda = get-python-app-deploy pkgs-with-cuda;
        optuna-env-with-cuda = get-python-app-optuna pkgs-with-cuda;
        with-cuda = webservice-with-cuda;
      }
    );
    
    # additional binaries we can run
    apps = {
      retrain-model = {
        type = "app";
        program = "${self.outputs.packages.${system}.default}/bin/retrain-model";
      };
      run-study = {
        type = "app";
        program = "${self.outputs.packages.${system}.optuna-env}/bin/retrain-model";
      };
    } //
    # CUDA is only supported on x86_64 linux
    (
      nixpkgs.lib.optionalAttrs
      (system == "x86_64-linux")
      {
        retrain-model-with-cuda = {
          type = "app";
          program = "${self.outputs.packages.${system}.with-cuda}/bin/retrain-model";
        };
        run-study-with-cuda = {
          type = "app";
          program = "${self.outputs.packages.${system}.optuna-env-with-cuda}/bin/retrain-model";
        };
      }
    );
    
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

    # integration testing
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
