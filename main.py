import os
from app import create_app, db
from app.models import Doc, Akteur

app = create_app(os.getenv('FLASK_CONFIG') or 'default')

@app.shell_context_processor
def make_shell_context():
    return dict(db=db, Doc=Doc, Akteur=Akteur)

@app.cli.command()
def test():
    """Run the unit tests."""
    import pytest
    pytest.main(["-x", "tests"])
