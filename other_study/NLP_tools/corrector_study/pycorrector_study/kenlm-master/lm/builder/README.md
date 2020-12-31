Dependencies
============

Boost >= 1.42.0 is required.  

For Ubuntu,
```bash
sudo apt-get install libboost1.48-all-dev
```

Alternatively, you can download, compile, and install it yourself:

```bash
wget http://sourceforge.net/projects/boost/files/boost/1.52.0/boost_1_52_0.tar.gz/download -O boost_1_52_0.tar.gz
tar -xvzf boost_1_52_0.tar.gz
cd boost_1_52_0
./bootstrap.sh
./b2
sudo ./b2 install
```

Local install options (in a user-space prefix directory) are also possible. See http://www.boost.org/doc/libs/1_52_0/doc/html/bbv2/installation.html.


Building
========

```bash
bjam
```
Your distribution might package bjam and boost-build separately from Boost.  Both are required.   

Usage
=====

Run
```bash
$ bin/lmplz
```
to see command line arguments

Running
=======

```bash
bin/lmplz -o 5 <text >text.arpa
```
