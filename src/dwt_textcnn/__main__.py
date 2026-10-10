"""Allow ``python -m dwt_textcnn`` to invoke the command line interface."""

from dwt_textcnn.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
