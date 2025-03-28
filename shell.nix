{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python and build tools
    python313
    python313Packages.build
    python313Packages.pip
    python313Packages.wheel
    python313Packages.setuptools
    
    # GitHub Actions local runner
    act
    
    # Development tools
    python313Packages.pytest
  ];

  shellHook = ''
    export PYTHONPATH="$PWD:$PYTHONPATH"
  '';
} 