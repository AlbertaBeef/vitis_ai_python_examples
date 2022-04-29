Requirements:

   The webserver.py requires the following package versions:
      Flask==1.1.2
      click==7.1.2
      Jinja2==2.11.2
      Werkzeug==1.0.1
      MarkupSafe==1.1.1
      itsdangerous==1.1.0
   Install with the following:
      pip3 install Flask==1.1.2 click==7.1.2 Jinja2==2.11.2 Werkzeug==1.0.1 MarkupSafe==1.1.1 itsdangerous==1.1.0

To start the webserver, type the following command:

   $ cd vitis_ai_python_examples/webserver
   $ python3 webserver.py


If running on Ultra96-V2, you may get the following error:

   OSError: [Errno 98] Address already in use

You will first need to stop the default webserver, with the following command:

   $ /etc/init.d/ultra96-startup-page.sh stop

