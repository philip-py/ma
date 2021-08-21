"""Compile static assets."""
from flask import current_app as app
from flask_assets import Bundle


def compile_static_assets(assets):
    """
    Compile stylesheets if in development mode.

    :param assets: Flask-Assets Environment
    :type assets: Environment
    """
    assets.auto_build = True
    assets.debug = False

    css = Bundle(
        "css/*.css",
        # filters="less,cssmin",
        output="gen/avantui.css",
        # extra={"rel": "stylesheet/less"},
    )

    js = Bundle(
        "js/*.js",
        output='gen/avantui.js'
    )

    assets.register("avantui_css", css)
    assets.register("avantui_js", js)
    if app.config["ENV"] == "development":
        css.build()
        js.build()
    return assets
