
import json
original = './dev-v2.0.json'
modified = './dev-v2.0modified.json'
remove = ["answers", "is_impossible","plausible_answers"]

with open(original) as f:
  data = json.load(f)
  c = data["data"]
  for element in c:
      for paragraph in element["paragraphs"]:
          for key in paragraph["qas"]:
              for item in remove:
                  if item in key:
                      del(key[item])
        
newObject = json.dumps(data)
# Writing to sample.json 
with open(modified, "w") as output: 
    output.write(newObject)
             
              
              
