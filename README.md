# Code for the GermEval 2024 KlarTextCoders submission 

Branch Emma

Hey Akilesh :)
If you want to put this on main, you can just take the LLM folder from here. Everything that belongs to 
the LLM analysis is in there. For the Readme, you can take the section below: 

# LLM/ Llama 
Our main analysis was predicting the number of statements and the statement spans. This was done using LLAMA-3-70B-Instruct. The model was run over resources provided by [Kisski](https://kisski.gwdg.de/), part of the Gwdg Göttingen. 

To reproduce our results, you have to:

 1. Request an API key from Kisski [here](https://kisski.gwdg.de/leistungen/2-02-llm-service/) by clicking on "Buchen" (Booking). The login requires an AcademicID. If you are a student or researcher from a German University you will be able to log in.

2. Add you API key to the script:

```python

from openai import OpenAI
import pandas as pd

# API configuration
api_key = "YOUR_API_KEY"  # Replace with your own API key

```

5. Execute the script and check your results!


