import click

from images import Download
from model1 import model1
from model2 import model2
from model3 import model3

#Define group. If no argument is passed, it will show the --help argument
@click.group()
@click.pass_context
def cli(ctx):
	pass

#Define download command
@cli.command()
def download():
	'''
	Call the Download function.
	'''
	Download()

#Define the infer command
@cli.command()
@click.option('-m', default="model3", help="Name fo model") #Argument of model
@click.option('-d', default="", help="Images directory") #Argument for images directory
def infer(m, d):
	'''
	This function will call the infer function in accordance to the model selected
	'''
	if m == 'model1':
		model1('infer', d)
	elif m ==  'model2':
		model2('infer', d)
	elif m == 'model3':
		model3('infer', d)
	else:
		print("The selected model does not exist. Use model1, model2 or model3.")

#Define the train command
@cli.command()
@click.option('-m', default="model3", help="Name fo model") #Argument of model
@click.option('-d', default="", help="Images directory") #Argument for images directory
def train(m, d):
	'''
	This function will call the train function in accordance to the model selected
	'''
	if m == 'model1':
		model1('train', d)
	elif m ==  'model2':
		model2('train', d)
	elif m == 'model3':
		model3('train', d)
	else:
		print("The selected model does not exist. Use model1, model2 or model3.")					 

#Define the test command
@cli.command()
@click.option('-m', default="model3", help="Name fo model") #Argument of model
@click.option('-d', default="", help="Images directory") #Argument for images directory
def test(m, d):
	'''
	This function will call the test function in accordance to the model selected
	'''
	if m == 'model1':
		model1('test', d)
	elif m ==  'model2':
		model2('test', d)
	elif m == 'model3':
		model3('test', d)
	else:
		print("The selected model does not exist. Use model1, model2 or model3.")					 


if __name__ == '__main__':
    cli(obj={})    