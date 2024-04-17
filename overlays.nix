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
  its-jointprobability = (
    final: prev:
    let
      # add the python library to python, without exposing it to the outside
      py-pkgs = (final.extend python-lib).python3Packages;
    in
    {
      its-jointprobability = py-pkgs.callPackage ./package.nix { };
    }
  );
}
