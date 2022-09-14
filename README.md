# MAD chess

## About The Project

*Project developed in the Autonomous Agents and Multi Agent Systems course, at IST (Lisboa)*

The objective of this project was to study the performance of a multi-agent approach to play chess based on supervised learning.

Inside the directory `code`, you can find the code scripts (in Python).

In the `report.pdf` file you can find:
* Chess rule simplifications
* Implementation concepts
* Learning process
  * Data collection
  * Model parameter decisions
* Agents
  * Implementation
  * Learning model
  * Behaviors
* Multi-agent system behavior
* Results

## Contributing

### Guidelines
* In order to create any agent, we follow the agent.py interface
* The agents dynamic together is in the file agentsplayground.py
* Use piecelists for your board, available in the staterepresentations folder			
* Do not touch either the interface or the playground without triple checking, it may break the code. 	

### Noise cancellation
Because we changed a lot of rules, the stockfish will produce massive amounts of noise. To cancel this you need to:
* Go to `~/.local/lib/python3.6/site-packages/chess/engine.py` and comment the line `LOGGER.exception("engine sent invalid ponder move")`.
* Write `pass` because python doesn't allow empy `excepts` 
			
```python
except ValueError:
	pass
	#LOGGER.exception("engine sent invalid ponder move")
```

## Contributors
This project was developed by Diogo Ramalho, Diogo Fernandes, Rafael Andrade and Tiago Oliveira.
