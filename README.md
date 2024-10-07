# GameGenerators

# Security Games Library

## Description

Security games are a framework in algorithmic game theory that model scenarios involving strategic interactions between adversarial agents. In these games, defender(s) aim to protect valuable targets from attacking player(s). The paper [Stackelberg Security Games: Looking Beyond a Decade of Success](https://www.ijcai.org/proceedings/2018/0775.pdf) provides a comprehensive overview of security games in the field, highlighting various models and strategies employed by researchers to model and test different game constructions. 

This library implements a flexible and customizable environment for generating security games, allowing users to define a wide range of parameters and scenarios. By leveraging real-world graphs generated through OSMnx, this library offers a realistic simulation of security scenarios, making it a valuable tool for our research group to standardize the creation of our games.

## Importance of the Library

Having a library that can generate security games with highly customized parameters is crucial for several reasons:

- **Flexibility**: Users can tailor the parameters to suit specific applications and scenarios, enhancing the relevance of the game simulations.
- **Realism**: By incorporating real graphs through OSMnx, the library enables the modeling of actual geographic areas, allowing for more accurate game-playing scenarios.
- **Research and Development**: This library serves as a foundation for exploring different strategies and game models in security games, and will be useful to standardize the process of game generation for our research group.

## Project Structure

```plaintext
security_games/
│
├── graphs/                  # Functions for creating and handling OSMnx (and eventually arbitrary) graphs
│
├── notebooks/               # Notebooks used in game development and testing
│
├── security_game/           # Core game models and logic
│   ├── game.py                   # Main security game implementation
│   ├── player.py                 # Player classes (attacker, defender)
│   ├── target.py                 # Target class
│
├── tests/                   # Unit tests for the library
│   ├── test_security_game.py     # Tests for security game functionality
│
└── utils/                   # Utility/visualization functions and helpers