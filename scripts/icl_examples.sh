# python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv-downsampleproj-e1-nose --conv-mode vicuna_v1_1 \
#     --query "Compute the results in the image." \
#     --image-file "images/icl-math/0.png,images/icl-math/1.png,images/icl-math/2.png" \
#     --answers "2+1=3,5+6=11"


python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv-downsampleproj-e1-nose --conv-mode vicuna_v1_1 \
    --query "Count number of animals in the image." \
    --image-file "images/icl-count/pandas.png,images/icl-count/dogs.png,images/icl-count/cats.jpg" \
    --answers "3 pandas,2 dogs"


python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv-downsampleproj-e1-nose --conv-mode vicuna_v1_1 \
    --query "Count number of animals in the image." \
    --image-file "images/icl-count/cats.jpg" 


python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv-downsampleproj-e1-nose --conv-mode vicuna_v1_1 \
    --query "What is the word in the image?" \
    --image-file "images/icl-ocr/underground.png,images/icl-ocr/congress.png,images/icl-ocr/soulomes.png" \
    --answers "Underground,Congress"


python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv-downsampleproj-e1-nose --conv-mode vicuna_v1_1 \
    --query "What is the word in the image?" \
    --image-file "images/icl-ocr/congress.png" 



python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv-downsampleproj-e1-nose --conv-mode vicuna_v1_1 \
    --query "" \
    --image-file "images/icl-animal/chinchilla.png,images/icl-animal/shiba.png,images/icl-animal/flamingo.png" \
    --answers "This is a chinchilla. They are mainly found in Chile.,This is a shiba. They are very popular in Japan."


python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv-downsampleproj-e1-nose --conv-mode vicuna_v1_1 \
    --query "" \
    --image-file "images/icl-dalle/dalle0.png,images/icl-dalle/dalle1.png,images/icl-dalle/dalle2.png" \
    --answers "A propaganda poster depicting a cat dressed as French emperor Napoleon holding a piece of cheese.,A pink room with a flamingo pool float."


python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv-downsampleproj-e1-nose --conv-mode vicuna_v1_1 \
    --query "" \
    --image-file "images/icl-french/violin.png,images/icl-french/snake.png,images/icl-french/swarm.png" \
    --answers "Les sanglots longs des violons de l’automne blessent mon coeur d’une langueur monotone.,Pour qui sont ces serpents qui sifflent sur vos têtes?"




### NEW TEMPLATE


python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv-downsampleproj-e1-nose --conv-mode vicuna_v1_1 \
    --query "Please describe the image." \
    --image-file "images/apple_ipod.png" 


python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv-downsampleproj-e1-nose --conv-mode vicuna_v1_1 \
    --query "<image>###A propaganda poster depicting a cat dressed as French emperor Napoleon holding a piece of cheese.###<image>###A pink room with a flamingo pool float.###<image>" \
    --image-file "images/icl-dalle/dalle0.png###images/icl-dalle/dalle1.png###images/icl-dalle/dalle2.png"


python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv-downsampleproj-e1-nose --conv-mode vicuna_v1_1 \
    --query "<image>###Les sanglots longs des violons de l’automne blessent mon coeur d’une langueur monotone.###<image>###Pour qui sont ces serpents qui sifflent sur vos têtes?###<image>" \
    --image-file "images/icl-french/violin.png###images/icl-french/snake.png###images/icl-french/swarm.png" 

# logo example
python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv+sharegpt-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "<image>###Google is a global technology leader best known for its search engine, and its suite of web-based services and products.###<image>###Microsoft is a global technology giant best known for products like Windows operating systems and the Office suite.###<image>###Apple Inc. is a multinational technology company known for its innovative hardware products like the iPhone, iPad, and Mac computers.###<image>" \
    --image-file "images/icl-logo/google.webp###images/icl-logo/microsoft.jpg###images/icl-logo/apple.jpg###images/icl-logo/nvidia.png"
python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-finetune-llava+lrv+sharegpt-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "<image>###Google; search engine.###<image>###Microsoft; operating systems.###<image>###Apple; iPhone.###<image>" \
    --image-file "images/icl-logo/google.webp###images/icl-logo/microsoft.jpg###images/icl-logo/apple.jpg###images/icl-logo/nvidia.png"


# conversation testing
python llava/eval/run_vila.py --model-path checkpoints/flanvicuna-7b-vflan-finetune-llava+lrv-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "What is the species of the puppy in the image?" \
    --image-file "images/puppy_and_kitten.webp" 
# The puppy in the image is a golden retriever.
python llava/eval/run_vila.py --model-path checkpoints/flanvicuna-7b-vflan-finetune-llava+lrv-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "What is the species of the puppy in the image?<image>###The puppy in the image is a golden retriever.###What does the truck in the image sell?<image>" \
    --image-file "images/puppy_and_kitten.webp###images/ice_cream_truck.png" 
# The truck in the image sells soft serve ice cream.
python llava/eval/run_vila.py --model-path checkpoints/flanvicuna-7b-vflan-finetune-llava+lrv-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "What is the species of the puppy in the image?<image>###The puppy in the image is a golden retriever.###What does the truck in the image sell?<image>###The truck in the image sells soft serve ice cream.###How many animals are there in the previous image?" \
    --image-file "images/puppy_and_kitten.webp###images/ice_cream_truck.png" 
# There are two animals in the previous image: a puppy and a kitten.
python llava/eval/run_vila.py --model-path checkpoints/flanvicuna-7b-vflan-finetune-llava+lrv-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "What is the species of the puppy in the image?<image>###The puppy in the image is a golden retriever.###What does the truck in the image sell?<image>###The truck in the image sells soft serve ice cream.###How many animals are there in the previous image?###There are two animals in the previous image: a puppy and a kitten.###Which two colors is the truck selling ice-cream in?" \
    --image-file "images/puppy_and_kitten.webp###images/ice_cream_truck.png" 
# The truck selling ice-cream is blue and white.


# two images input
python llava/eval/run_vila.py --model-path checkpoints/flanvicuna-7b-vflan-finetune-llava+lrv-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "How many animals are there in the two images? <image><image>" \
    --image-file "images/puppy_and_kitten.webp###images/golden_retriever.jpg" 

python llava/eval/run_vila.py --model-path checkpoints/flanvicuna-7b-vflan-finetune-llava+lrv-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "Take a look at the image. <image>###OK.###Take a look at the image. <image>###OK.###Which image has fewer animals?" \
    --image-file "images/puppy_and_kitten.webp###images/golden_retriever.jpg" 


python llava/eval/run_vila.py --model-path checkpoints/flanvicuna-7b-vflan-finetune-llava+lrv-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "How many images are there? \n<image>\n<image>###There are two images.###How many cats are there in the two images in total?" \
    --image-file "images/puppy_and_kitten.webp###images/golden_retriever.jpg" 
# There are two dogs in total, one in each image.
python llava/eval/run_vila.py --model-path checkpoints/flanvicuna-7b-vflan-finetune-llava+lrv-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "How many images are there? \n<image>\n<image>###There are two images.###How many dogs are there in the two images in total?###There are two dogs in total, one in each image.###How many cats are there in the two images in total?" \
    --image-file "images/puppy_and_kitten.webp###images/golden_retriever.jpg" 


python llava/eval/run_vila.py --model-path checkpoints/flanvicuna-7b-vflan-finetune-llava+lrv-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "This is the first image.\n<image>###OK.###This is the second image.\n<image>###OK.###What do the two images have in common?" \
    --image-file "images/puppy_and_kitten.webp###images/golden_retriever.jpg" 
# The two images have the same subjects – a dog and a cat. The main difference is the context in which the subjects are presented. In the first image, the dog and the cat are standing next to each other in a grassy field, while in the second image, the dog is sitting on a couch.

python llava/eval/run_vila.py --model-path checkpoints/flanvicuna-7b-vflan-finetune-llava+lrv-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "This is the first image.\n<image>###OK.###This is the second image.\n<image>###OK.###In which image there is a kitten?" \
    --image-file "images/puppy_and_kitten.webp###images/golden_retriever.jpg" 
# The kitten is in the first image.



# reasoning examples
python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-vflan-finetune-llava+lrv-downsampleprojse-e1-nose-tuneffn --conv-mode vicuna_v1_1 \
    --query "How many animals are there in the image?" \
    --image-file "images/puppy_and_kitten.webp" 
python llava/eval/run_vila.py --model-path checkpoints/vicuna-13b-vflan-finetune-llava+lrv-downsampleprojse-e1-nose-tuneffn --conv-mode vicuna_v1_1 \
    --query "How many animals are there in the image?\n<image>###There are two animals in the image: a dog and a cat.###If we take multiply the number of animals by 10 and add 5, how many is that?" \
    --image-file "images/puppy_and_kitten.webp" 




python llava/eval/run_vila.py --model-path checkpoints/llama2-7b-mmc4sub+coyoaccum-finetune-dualflan-linearclip-e1-nose --conv-mode vicuna_v1_1 \
    --query "<image>Should you honk at the pedestrians crossing the road?" \
    --image-file "images/av/av1.png" 



# new llama2 model
python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1_nosys \
    --query "<image>\n###Les sanglots longs des violons de l’automne blessent mon coeur d’une langueur monotone.###<image>\n###Pour qui sont ces serpents qui sifflent sur vos têtes?###<image>\n" \
    --image-file "images/icl-french/violin.png###images/icl-french/snake.png###images/icl-french/swarm.png" 


python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1_nosys \
    --query "<image>\n###The current population of Australia is 26 million people.###<image>\n###Deutschland hat 83 Millionen Einwohner.###<image>\n###Argentina tiene 46 millonesde habitantes.###<image>\n" \
    --image-file "images/icl-multi-lang/0.png###images/icl-multi-lang/1.png###images/icl-multi-lang/2.png###images/icl-multi-lang/3.png" 
# 7b: Die Eiffelturm hat 130 Meter hoch.
# 13: Francia tiene 118 millones de habitantes.
python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1_nosys \
    --query "<image>\n###The current population of Australia is 26 million people.###<image>\n###Deutschland hat 83 Millionen Einwohner.###<image>\n###中国有13亿人口.###<image>\n###Argentina tiene 46 millonesde habitantes.###<image>\n" \
    --image-file "images/icl-multi-lang/0.png###images/icl-multi-lang/1.png###images/icl-multi-lang/great-wall.jpg###images/icl-multi-lang/2.png###images/icl-multi-lang/3.png" 



python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1_nosys \
    --query "<image>\n###The company is famous for its search engine.###<image>\n###The company is famous for the operating systems.###<image>\n###The company is famous for iPhone and Mac.###<image>" \
    --image-file "images/icl-logo/google.webp###images/icl-logo/microsoft.jpg###images/icl-logo/apple.jpg###images/icl-logo/nvidia.png"
# The company is famous for its graphics processing units (GPUs) and other computer components.

python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\nCan you explain the meme?" \
    --image-file "images/meme1.png" 

# counting
python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1_nosys \
    --query "<image>\n###pandas: 3###<image>\n###dogs: 2###<image>\n" \
    --image-file "images/icl-count/pandas.png###images/icl-count/dogs.png###images/icl-count/giraffs.png"



python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\nWhat is unusual about this image?" \
    --image-file "images/av/chair.png" 
# The unusual aspect of this image is that a chair is flying through the air on a highway, seemingly coming out of the back of a truck. This is an unexpected and unusual sight, as chairs are not typically transported in this manner. The scene also includes other vehicles, such as cars and a motorcycle, driving on the highway, which adds to the overall peculiarity of the situation.

python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\nWhat is unusual about this image?###The unusual aspect of this image is that a chair is flying through the air on a highway, seemingly coming out of the back of a truck.###What should the driver do in this case?" \
    --image-file "images/av/chair.png" 
# If you encounter this situation, you should immediately stop your vehicle and move to a safe distance from the truck and the flying chair. It is essential to avoid any potential hazards and contact the authorities to report the incident and ensure the safety of everyone involved.


python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-clip336-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\nDescribe the image in details." \
    --image-file "images/dalle3/leaf.avif" 


python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-clip336-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\nDescribe the image in details.###A 2D animation of a folk music band composed of anthropomorphic autumn leaves, each playing traditional bluegrass instruments, amidst a rustic forest setting dappled with the soft light of a harvest moon.###<image>\nDescribe the image in details.###A vast landscape made entirely of various meats spreads out before the viewer. tender, succulent hills of roast beef, chicken drumstick trees, bacon rivers, and ham boulders create a surreal, yet appetizing scene. the sky is adorned with pepperoni sun and salami clouds.###<image>\nDescribe the image in details." \
    --image-file "images/dalle3/leaf.jpeg###images/dalle3/food.jpeg###images/dalle3/potatoking.jpeg" 


python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-clip336-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\n###A 2D animation of a folk music band composed of anthropomorphic autumn leaves, each playing traditional bluegrass instruments, amidst a rustic forest setting dappled with the soft light of a harvest moon.###<image>\n###A vast landscape made entirely of various meats spreads out before the viewer. tender, succulent hills of roast beef, chicken drumstick trees, bacon rivers, and ham boulders create a surreal, yet appetizing scene. the sky is adorned with pepperoni sun and salami clouds.###<image>\n###A vibrant yellow banana-shaped couch sits in a cozy living room, its curve cradling a pile of colorful cushions. on the wooden floor, a patterned rug adds a touch of eclectic charm, and a potted plant sits in the corner, reaching towards the sunlight filtering through the window.###<image>\n" \
    --image-file "images/dalle3/leaf.jpeg###images/dalle3/food.jpeg###images/dalle3/banana.jpeg###images/dalle3/potatoking.jpeg" 


python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-clip336-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\n###A 2D animation of a folk music band composed of anthropomorphic autumn leaves, each playing traditional bluegrass instruments, amidst a rustic forest setting dappled with the soft light of a harvest moon.###<image>\n###Tiny potato kings wearing majestic crowns, sitting on thrones, overseeing their vast potato kingdom filled with potato subjects and potato castles.###<image>\n###A vibrant yellow banana-shaped couch sits in a cozy living room, its curve cradling a pile of colorful cushions. on the wooden floor, a patterned rug adds a touch of eclectic charm, and a potted plant sits in the corner, reaching towards the sunlight filtering through the window.###<image>\n" \
    --image-file "images/dalle3/leaf.jpeg###images/dalle3/potatoking.jpeg###images/dalle3/banana.jpeg###images/dalle3/food.jpeg" 


python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-clip336-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\nCan you explain the meme?" \
    --image-file "images/dalle3/oai_meme.jpeg" 


python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1_nosys \
    --query "<image>\n###Home to the best burgers and fried chicken.###<image>\n###Home to unbeatable fish and chips.###<image>\n###Home to outstanding ramen. ###<image>" \
    --image-file "images/icl-hometo/im1.png###images/icl-hometo/im2.png###images/icl-hometo/im3.png###images/icl-hometo/im4.png"


python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-clip336-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\nCan you explain the meme?" \
    --image-file "images/adobe-meme/1.jpg" 



### FINAL

python llava/eval/run_vila.py --model-path ~/models/llava/llava-v1.5-13b-patched/ --conv-mode vicuna_v1_1 \
    --query "<image>\n###Les sanglots longs des violons de l’automne blessent mon coeur d’une langueur monotone.###<image>\n###Pour qui sont ces serpents qui sifflent sur vos têtes?###<image>\n" \
    --image-file "images/icl-french/violin.png###images/icl-french/snake.png###images/icl-french/swarm.png"


python llava/eval/run_vila.py --model-path ~/models/llava/llava-v1.5-13b-patched/ --conv-mode vicuna_v1_1 \
    --query "<image>\n###pandas: 3###<image>\n###dogs: 2###<image>\n" \
    --image-file "images/icl-count/pandas.png###images/icl-count/dogs.png###images/icl-count/giraffs.png" --pad


python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\n###Home to the best burgers and fried chicken.###<image>\n###Home to unbeatable fish and chips.###<image>\n" \
    --image-file "images/icl-hometo/im1.png###images/icl-hometo/im2.png###images/icl-hometo/im3.png"

# Home to the best ramen in town.
# Home to the best sushi and sake.
# Home to the best ramen and sake.

python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\n###The current population of Australia is 26 million people.###<image>\n###Germany has 83 million inhabitants.###<image>\n" \
    --image-file "images/icl-multi-lang/0.png###images/icl-multi-lang/1.png###images/icl-multi-lang/3.png"

# The current population of France is 120 million people.


python llava/eval/run_vila.py --model-path checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\n###The current population of Australia is 26 million people.###<image>\n###Germany has 83 million inhabitants.###<image>\n###Argentina tiene 46 millones de habitantes.###<image>\n" \
    --image-file "images/icl-multi-lang/0.png###images/icl-multi-lang/1.png###images/icl-multi-lang/2.png###images/icl-multi-lang/3.png"


python llava/eval/run_vila.py --model-path ~/models/llava/llava-v1.5-13b-patched/ --conv-mode vicuna_v1_1 \
    --manual_prompt "I like reading <image>\n, my favourite play is Hamlet. I also like <image>\n, my favorite book is" --image-file "images/gpt4/shakespeare.png###images/gpt4/obama.png"