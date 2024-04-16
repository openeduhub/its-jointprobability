{
  lib,
  nix-filter,
  its-data-overlay,
}:
rec {
  default = its-jointprobability;

  # add the python library and its related python libraries
  python-lib = lib.composeExtensions its-data-overlay (
    final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (python-final: python-prev: {
          its-jointprobability = python-final.callPackage ./python-lib.nix { inherit nix-filter; };
          its-jointprobability-with-oputuna = python-final.callPackage ./python-lib.nix {
            inherit nix-filter;
            withOptuna = true;
          };
        })
      ];
    }
  );

  # add the standalone python application (without also adding the python
  # library)
  its-jointprobability = lib.composeExtensions its-data-overlay (
    final: prev:
    let
      py-pkgs = final.python3Packages;
      its-jointprobability = py-pkgs.callPackage ./python-lib.nix { inherit nix-filter; };
    in
    {
      its-jointprobability = py-pkgs.callPackage ./package.nix { inherit its-jointprobability; };
    }
  );
}
