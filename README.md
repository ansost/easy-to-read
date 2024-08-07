# Code for the GermEval 2024 KlarTextCoders submission 

Hey Akilesh :)

If you want to put this on main, you can just take the LLM folder from here. Everything that belongs to 
the LLM analysis is in there. For the Readme, you can take this section below: 

# LLM/ Llama 
The LLM for our analysis was run over resources provided by [Kisski](https://kisski.gwdg.de/), part of Gwdg Göttingen. We report all our prompts used and utilize seeds to ensure replicability. 
> Note that the seed feature is stil in its beta. For more information on seeds, see OpenAI's [documentation](https://platform.openai.com/docs/api-reference/chat/create).

To reproduce our results, you have to:

 1. Request an API key from Kisski [here](https://kisski.gwdg.de/leistungen/2-02-llm-service/) by clicking on "Buchen" (Booking).

 2. You will be redirected to an AcademicID login. If you are a student or researcher from a German University you will be able to log in with your university credentials or create a new account and aqcuire the API key.

4. Add you API key to the script:
> ``LLM/API_prompts_num_statements.py`` or ``LLM/API_prompts_statement_spans.py``, depending on what prompt you want.

```python

from openai import OpenAI
import pandas as pd

# API configuration
api_key = "YOUR_API_KEY"  # Replace with your own API key

```

5. Execute the script and check your results!


We also experimented with Llama for subcases from the annotation guidelines with manual prompting over [Huggingchat](https://huggingface.co/chat/). You can find the prompts used and an example answer in ``Llama/prompts_subtasks.md``


