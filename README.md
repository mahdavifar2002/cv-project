# Fundamentals of 3D Computer Vision: G0 Project

  | `First Name` | `Last Name` | `Student Number` |
  |:------------:|:-----------:|:----------------:|
  | Ali | Mahdavifar | 98106072  |
  | Iman | Alipour | 98102024 |
  


## Requirements
In order to install the requirements, run
```
pip install -r requirements.txt
```

You need to install npm for setting up the front-end server.
```
npm install http-server -g
```

## Usage

### Front-end Server

In order to run the front-end server, run the command bellow in `./webxr` directory.

```
http-server -S -C ./keys/fullchain.pem -K ./keys/privkey.pem -p 5001
```

You need to have valid SSL keys for running the server. You can create them using certbot.

### Back-end Server

In order to run the back-end server, run the command bellow in `./flask` directory.

```
python3 app.py
```