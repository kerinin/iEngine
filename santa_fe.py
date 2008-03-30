#! /usr/bin/env pythonimport sys, os, time, datetime, getopt
def parsePath(path):	# check if file exists	if os.path.isfile(path):		parse_file = file(path,'r')		print path		data = parseFile( parse_file,path )	elif os.path.isdir(path):		for sub_path in os.listdir(path):			data = parsePath(os.path.join(path,sub_path))
	return data			def parseFile(parse_file,path):	data = array()	# parse file	for line in parse_file.readlines():		data.append( line.split(" ") )	return data
def getData(dataset=''):
	return parsePath('Santa_Fe_Competition'+dataset)	 		
