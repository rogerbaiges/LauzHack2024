from controller import Controller

controller = Controller()

while True:
	prompt = input("Prompt: ")
	print(controller.run(prompt, image_path='./uploads/cows.png'))