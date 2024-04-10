{
  lib,
  system,
  mkShell,
  python3,
  pyright,
  nix-template,
  nix-init,
  nix-tree,
  nix-filter,
}:
mkShell {
  packages = [
    (python3.withPackages (
      py-pkgs:
      with py-pkgs;
      [
        black
        pyflakes
        isort
        pylint
        ipython
        mypy
      ]
      # pandas-stubs appears to be broken on darwin systems
      ++ (lib.lists.optionals (system != "x86_64-darwin" || system != "aarch64-darwin") [
        py-pkgs.pandas-stubs
      ])
      ++ (py-pkgs.callPackage ./python-lib.nix {
        inherit nix-filter;
        withOptuna = true;
      }).propagatedBuildInputs
    ))
    pyright
    nix-template
    nix-init
    nix-tree
  ];
}
