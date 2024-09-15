from asgiref.wsgi import WsgiToAsgi
from application import application

asgi_app = WsgiToAsgi(application)