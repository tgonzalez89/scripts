import re


def filter_cards(input_text):
    filtered_lines = []

    # Regular expression pattern to match lines like "3x Archweaver"
    pattern = r"^\s*(\d+)x (.+)$"

    # Split the input text into lines
    lines = input_text.split("\n")

    # Iterate over each line
    for line in lines:
        # Use regex to match the pattern
        match = re.match(pattern, line)
        if match:
            # Get the matched line (removing leading and trailing spaces)
            filtered_line = f"{match.group(1)} {match.group(2)}"
            filtered_lines.append(filtered_line)

    return filtered_lines


if __name__ == "__main__":
    # Input text as given
    input_text = """
Creature (9)

    1x Arbor Elf
    1x Elder of Laurels
    1x Gigantosaurus
    1x Greenweaver Druid
    1x Leatherback Baloth
    1x Molimo, Maro-Sorcerer
    1x Norwood Warrior
    1x Plated Crusher
    1x Sakura-Tribe Elder

Instant (5)

    1x Awaken the Bear
    1x Blossoming Defense
    1x Fog
    1x Ranger's Guile
    1x Regenerate

Sorcery (2)

    1x Adventurous Impulse
    1x Overwhelming Stampede

Enchantment (1)

    1x Fertile Ground


Creature (11)

    1x Bloodrage Brawler
    1x Bogardan Dragonheart
    1x Borderland Marauder
    1x Dragon Hatchling
    1x Fanatical Firebrand
    1x Flameblast Dragon
    1x Guttersnipe
    1x Lightning Shrieker
    1x Minotaur Aggressor
    1x Monastery Swiftspear
    1x Siegebreaker Giant

Instant (8)

    1x Enrage
    1x Fireblast
    1x Gut Shot
    1x Magma Jet
    1x Needle Drop
    1x Searing Blood
    1x Temur Battle Rage
    1x Titan's Strength

Sorcery (7)

    1x Blaze
    1x Cosmotronic Wave
    1x Doublecast
    1x Fireball
    1x Forked Bolt
    1x Rolling Thunder
    1x Stone Rain


Creature (16)

    1x Aerial Responder
    1x Alabaster Mage
    1x Atalya, Samite Master
    1x Aysen Crusader
    1x Captain of the Watch
    1x Doomed Traveler
    1x Elite Vanguard
    1x Everdawn Champion
    1x Fairgrounds Warden
    1x Icatian Javelineers
    1x Icatian Lieutenant
    1x Lena, Selfless Champion
    1x Militia Bugler
    1x Perimeter Captain
    1x Sun Sentinel
    1x Veteran Armorer

Enchantment (7)

    1x Crusade
    1x First Response
    1x Hope Against Hope
    1x Intangible Virtue
    1x Mobilization
    1x Seal Away
    1x Silkwrap

Instant (4)

    1x Bandage
    1x Make a Stand
    1x Recruit the Worthy
    1x Riot Control

Sorcery (1)

    1x Sunlance

Land (1)

    1x Memorial to Glory


Creature (12)

    1x Bloodmist Infiltrator
    1x Bone Picker
    1x Demon of Catastrophes
    1x Diregraf Ghoul
    1x Drakestown Forgotten
    1x Eternal Taskmaster
    1x Gorging Vulture
    1x Graveblade Marauder
    1x Gravewaker
    1x Hypnotic Specter
    1x Nezumi Bone-Reader
    1x Soldevi Adnate

Sorcery (4)

    1x Consume Spirit
    1x Diabolic Tutor
    1x Dread Return
    1x Raise Dead

Instant (2)

    1x Grim Return
    1x Sacrifice

Enchantment (3)

    1x Curse of Death's Hold
    1x Nettlevine Blight
    1x Weakness

Maybeboard
Creature (12)

    1x Cathartic Adept
    1x Cephalid Broker
    1x Cryptic Serpent
    1x Geist of the Archives
    1x Hapless Researcher
    1x Mizzium Meddler
    1x Nightveil Sprite
    1x Renegade Doppelganger
    1x Sigiled Starfish
    1x Wall of Frost
    1x Wall of Runes
    1x Watcher for Tomorrow

Instant (9)

    1x Boomerang
    1x Censor
    1x Desertion
    1x Dissolve
    1x Impulse
    1x Opt
    1x Peek
    1x Quicken
    1x Rescind

Enchantment (6)

    1x Capture Sphere
    1x Control Magic
    1x Jace's Sanctum
    1x Lay Claim
    1x Ophidian Eye
    1x Oracle's Insight

Sorcery (8)

    1x Careful Study
    1x Contingency Plan
    1x Flux
    1x Pore Over the Pages
    1x Rise from the Tides
    1x Set Adrift
    1x Shimmer of Possibility
    1x Stream of Thought

"""

    # Call the filtering function
    filtered_cards = filter_cards(input_text)

    # Print the filtered lines
    for card in filtered_cards:
        print(card)
