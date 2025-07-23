import random
import json
import time

# Synonyms and variations of keywords
image_or_photo = ["An image of", "A photo of"]

turtle_variants = [
    "a sea turtle", "a green turtle", "a marine turtle", "a solitary turtle", "a lone turtle",
    "a turtle", "an ocean turtle", "an aquatic turtle", "a juvenile sea turtle", "a swimming turtle"
]
dolphin_variants = [
    "a dolphin", "a marine dolphin", "a playful dolphin", "a solitary dolphin",
    "an ocean dolphin", "an aquatic dolphin", "a juvenile dolphin", "a swimming dolphin", "a spotted dolphin"
]
satellite_variants = [
    "satellite image", "aerial view", "drone-like view"
]

ocean_contexts = [
    "in the vast ocean", "in the deep sea", "in the ocean", "in the sea",
    "surrounded by currents", "in still marine waters", "in a remote open sea", "swimming in the open ocean",
    "within open sea", "in a remote oceanic area", "in the water"
]

actions = [
    "drifts", "moves",
    "is captured swimming", "floats", "navigates", "cruises",
    "swims through the waves", "swims"
]

ocean_contexts_clean = [
    "An image of the open sea", "An image of the ocean's surface",
    "An image of marine waters", "An image of the ocean", "An image of the marine surface", "An image of the sea", "An image of the water"
]

'''ocean_contexts_with_trash =[
    "An image of garbage in the ocean", "A photo of garbage in the sea",
    "An image featuring of trash in the ocean", "A photo of debris in the ocean", "An image of plastic waste in the sea", "An image of marine litter"
]'''

ocean_contexts_with_trash = [
    "A garbage in the ocean", "A garbage in the sea", "A debris in the ocean", "A plastic waste in the sea", "A marine litter"
]


def shuffle(dynamic_random):
    dynamic_random.shuffle(ocean_contexts_with_trash)
    dynamic_random.shuffle(ocean_contexts_clean)
    dynamic_random.shuffle(actions)
    dynamic_random.shuffle(ocean_contexts)
    dynamic_random.shuffle(satellite_variants)
    dynamic_random.shuffle(turtle_variants)
    dynamic_random.shuffle(image_or_photo)

def generate_dolphine_sentence():
    dynamic_random = random.Random(time.time())
    shuffle(dynamic_random)
    dolphine = dynamic_random.choice(dolphin_variants)
    context = dynamic_random.choice(ocean_contexts)
    action = dynamic_random.choice(actions)
    return f"{dolphine} {action} {context}."


def generate_positive_sentence():
    dynamic_random = random.Random(time.time())
    shuffle(dynamic_random)
    turtle = dynamic_random.choice(turtle_variants)
    context = dynamic_random.choice(ocean_contexts)
    action = dynamic_random.choice(actions)
    return f"{turtle} {action} {context}."


def generate_negative_sentence(include_trash=False):
    dynamic_random = random.Random(time.time())
    shuffle(dynamic_random)
    if include_trash:
        context = dynamic_random.choice(ocean_contexts_with_trash)
    else:
        context = dynamic_random.choice(ocean_contexts_clean)
    return f"{context}"

    

