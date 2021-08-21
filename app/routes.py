"""Routes for parent Flask app."""
from flask import render_template
from flask import current_app as app
# from app.src.d02_web.graphs import get_populism_graph


@app.route('/')
def home():
    """Landing page."""
    return render_template(
        'index.jinja2',
        title='MdB Monitor',
        description='Making Democracy more transparent',
        template='home-template',
        # body="This is a homepage served with Flask."
    )

# @app.route('/essay')
# def essay():
#     """Impressum Page."""
#     return render_template(
#         'essay.jinja2',
#         title='Essay',
#         description='Essay & Blog',
#         template='essay-template',
#     )

# @app.route('/essay/<int:id>')
# def blogs(id):
#     """Loads an article"""
#     if id == 1:
#         g = get_populism_graph()
#     else:
#         g = None

#     return render_template(
#         'article.jinja2',
#         title=f'Article {id}',
#         description='Essay & Blog',
#         template='article-template',
#         id=id,
#         graph=g
#     )

# @app.route('/contact')
# def contact():
#     """Impressum Page."""
#     return render_template(
#         'contact.jinja2',
#         title='Kontakt',
#         description='Impressum Page',
#         template='contact-template',
#     )
