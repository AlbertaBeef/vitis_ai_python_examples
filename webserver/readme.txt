
To start the webserver, type the following command:

   $ cd vitis_ai_python_examples/webserver
   $ python3 webserver.py


If running on Ultra96-V2, you may get the following error:

   OSError: [Errno 98] Address already in use

You will first need to stop the default webserver, with the following command:

   $ /etc/init.d/ultra96-startup-page.sh stop

