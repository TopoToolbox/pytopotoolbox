# Generation Sphinx Documentation

Before being able to generate the Docs, make sure to install all the python dependencies: `pip install -r requirements.txt` and pandoc. To generate the Sphinx documentation HTML page, run the following commands (Linux):

```bash
cd /path/to/topotoolbox/docs/
make clean
make html
```

then open the `pytopotoolbox/docs/_build/html/index.html` file in your browser.
