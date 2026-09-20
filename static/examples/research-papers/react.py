"""Complete ReAct runner with real tools and an optional model endpoint.
Default: deterministic test fixture checks the runner, not language reasoning.
For model mode set REACT_ENDPOINT (full chat-completions URL), REACT_MODEL and
optionally REACT_API_KEY. Standard-library HTTP only; tools use a local corpus.
"""
import json
import os
import re
import urllib.request

CORPUS = {
    'France': 'France is a country in Europe. Its capital is Paris.',
    'Paris': 'Paris is the capital of France. The river Seine flows through Paris.',
    'Germany': 'Germany is a country in Europe. Its capital is Berlin.',
    'Berlin': 'Berlin is the capital of Germany. The river Spree flows through Berlin.',
}
SYSTEM = """You answer questions using a local encyclopaedia.
Return a brief next-step plan, then exactly one action in this format:
Plan: <brief next step>
Action: Search[entity] OR Lookup[keyword] OR Finish[answer]
Search opens a page. Lookup finds a sentence on the most recently opened page.
Observations come from the program; do not invent them.
Example:
Question: Which river crosses the capital of Germany?
Plan: Find Germany's capital.
Action: Search[Germany]
Observation: Germany is a country in Europe. Its capital is Berlin.
Plan: Find the river in Berlin.
Action: Search[Berlin]
Observation: Berlin is the capital of Germany. The river Spree flows through Berlin.
Plan: The retrieved page answers the question.
Action: Finish[Spree]
"""

class Environment:
    def __init__(self): self.page = None
    def execute(self, action, argument):
        if action == 'Search':
            self.page = next((k for k in CORPUS if k.lower() == argument.lower()), None)
            return CORPUS[self.page] if self.page else 'Page not found. Available: ' + ', '.join(CORPUS)
        if action == 'Lookup':
            if self.page is None: return 'Open a page using Search first.'
            matches = [s.strip() for s in CORPUS[self.page].split('.') if argument.lower() in s.lower()]
            return '. '.join(matches) or 'No matching sentence on this page.'
        raise ValueError('Unsupported tool')

def model_policy(history):
    payload = json.dumps({'model':os.environ['REACT_MODEL'], 'messages':history,
                          'temperature':0, 'stop':['\nObservation:']}).encode()
    headers = {'Content-Type':'application/json'}
    if os.getenv('REACT_API_KEY'): headers['Authorization'] = 'Bearer '+os.environ['REACT_API_KEY']
    request = urllib.request.Request(os.environ['REACT_ENDPOINT'], data=payload, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)['choices'][0]['message']['content']

def fixture_policy(history):
    # Deliberately explicit fixture, used only to test action execution end to end.
    observations = [m['content'] for m in history if m['content'].startswith('Observation:')]
    if not observations: return 'Plan: Find the capital.\nAction: Search[France]'
    if len(observations)==1: return 'Plan: Inspect the capital page.\nAction: Search[Paris]'
    if len(observations)==2: return 'Plan: Locate the river sentence.\nAction: Lookup[river]'
    return 'Plan: Answer from the retrieved sentence.\nAction: Finish[Seine]'

def run(question, policy, max_steps=8):
    environment = Environment()
    history = [{'role':'system','content':SYSTEM}, {'role':'user','content':'Question: '+question}]
    for _ in range(max_steps):
        text = policy(history)
        print(text)
        history.append({'role':'assistant','content':text})
        # Full line parsing prevents executing arbitrary model-written code.
        actions = re.findall(r'^Action: (Search|Lookup|Finish)\[([^\n]*)\]$', text, flags=re.M)
        if len(actions) != 1:
            observation = 'Invalid action. Return exactly one Search, Lookup or Finish action.'
        else:
            action, argument = actions[0]
            if action == 'Finish': return argument, history
            observation = environment.execute(action,argument)
        print('Observation:', observation)
        history.append({'role':'user','content':'Observation: '+observation})
    raise RuntimeError('Step budget exhausted without a final answer')

if __name__ == '__main__':
    policy = model_policy if os.getenv('REACT_ENDPOINT') else fixture_policy
    print('Mode:', 'model' if policy is model_policy else 'deterministic runner test')
    answer, trace = run('Which river crosses the capital of France?',policy)
    print('Answer:',answer)
    if policy is fixture_policy: assert answer=='Seine'
    with open('react-trace.json','w') as file: json.dump(trace,file,indent=2)
