from controller import Controller

controller = Controller()

while True:
	prompt = input("Prompt: ")
	image_path = input("Image path: ")
	print(controller.run(prompt, image_path='./uploads/cows.png'))