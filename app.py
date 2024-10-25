from flask import Flask, render_template, request
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# from pyngrok import ngrok, conf
# conf.get_default().auth_token = ""

# public_url = ngrok.connect(8888).public_url
# print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 5000))

# # Update any base URLs to use the public ngrok URL
# app.config["BASE_URL"] = public_url

app = Flask(__name__)

# Load the Stable Diffusion model
model_id = "cuongdev/vtthuc3"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cpu")

# Change the scheduler to DPMSolverMultistepScheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    negative_prompt = request.form['negative_prompt']
    guidance_scale = float(request.form['guidance_scale'])
    num_images = int(request.form['num_images'])
    seed = -1

    generator = torch.manual_seed(seed) if seed != -1 else None
    images = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        guidance_scale=guidance_scale, 
        generator=generator,
        num_images_per_prompt=num_images
    ).images

    # Save generated images and return paths
    image_paths = []
    for i, image in enumerate(images):
        image_path = f'static/generated_image_{i}.png'
        image.save(image_path)
        image_paths.append(image_path)

    return {"image_paths": image_paths}

if __name__ == '__main__':
    app.run(debug=True)
