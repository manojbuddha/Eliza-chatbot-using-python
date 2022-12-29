# Importing the required libraries
import re
import random
from string import punctuation
import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger', quiet=True)


# This function is used to load the pre-defined rules Eliza uses to spot keywords along with decomposition
# rules(regular expression patterns) and rearrangement rules (replies).
def load_dict():
    with open("chatbot_rules.json", "r") as chatbot_rules_file:
        chatbot_rules = json.load(chatbot_rules_file)

    return chatbot_rules


# Defining required global variables
global RANK
global key_stack
global REPLIES
global REPLY_LENGTH

REPLY_LENGTH = 0
REPLIES = []
RANK = 0
key_stack = []

# Loading our pre-defined keyword dictionary
null = 'null'
chatbot_rules = load_dict()


# Function to remove punctuations and convert multiple space to single space
def clean_text(text):
    text = re.sub(rf"[{punctuation}]", "", text)
    text = re.sub(r"\W+", " ", text)
    return text


# It takes keyword as input and gets rank for the keyword from the pre-defined dictionary.Based on the rank it will
# arrange the keyword in a list. If the rank is higher, the keyword is palced in the begining of the list.
def get_rank(token):
    global RANK
    global key_stack
    if chatbot_rules.get(token) is not None:
        rank = chatbot_rules.get(token).get('RANK')
        if rank is None:
            rank = 0
        if RANK > int(rank):
            key_stack = key_stack + [token]
        else:
            key_stack = [token] + key_stack
            RANK = int(rank)
        return "TOKEN_FOUND"
    else:
        return "TOKEN_NOT_FOUND"


# Takes token and user input , applies transformation (pronoun, 'AM to ARE' or other similar conversion) ,
# then apply decomposition and rearrangement rules for the keyword and returns Eliza's response.
def get_reply(token, text):
    global REPLIES
    global REPLY_LENGTH

    transformation_flag = False
    decoposition_rule = list(chatbot_rules.get(token).keys())

    if decoposition_rule is None:
        return "DECOMPOSITION_RULE_NOT_FOUND"

    if chatbot_rules.get(token).get("TRANSFORMATION") not in [None, 'null']:
        transformation_flag = True

    words = text.split()

    for i in range(0, len(words)):
        if words[i] == token and transformation_flag:
            words[i] = chatbot_rules.get(token).get("TRANSFORMATION")
            continue
        if token in ["I", "YOU"] and words[i] in ["I", "YOU"]:
            continue
        if words[i] != token:
            pronoun = pronoun_transform(words[i])

            if pronoun is not None:
                words[i] = pronoun
    text = " ".join(words)

    for rule in decoposition_rule:

        if rule == 'RANK':
            continue
        if rule == 'TRANSFORMATION':
            continue
        rule_t = rule.split()
        pattern = "(" + rule_t[0] + ")"
        for i in range(0, len(rule_t) - 1):
            if rule_t[i] != '0' and rule_t[i + 1] != 0:
                pattern = pattern + " " + "(" + rule_t[i + 1] + ")"
            else:
                pattern = pattern + "(" + rule_t[i + 1] + ")"

        pattern = pattern.replace('0', r'[a-zA-Z0-9 _]*').strip()

        match = re.match(pattern, " " + text.upper() + " ")

        if match is None:
            continue
        reply = chatbot_rules.get(token).get(rule)[0]

        if "=" in reply:
            reply = get_reply(reply.replace("=", ""), text)
            return reply

        if reply == 'NEWKEY':
            return "NEXT_KEY"

        if match is not None:
            match = match.groups()
            random.shuffle(chatbot_rules.get(token).get(rule))
            for word in reply.split():
                if word.strip().isnumeric():
                    reply = reply.replace(word.strip(), match[int(word) - 1])

            if len(rule) > REPLY_LENGTH:
                REPLIES = [reply] + REPLIES
            else:
                REPLIES.append(reply)
            return reply
        else:
            reply = "MATCH_NOT_FOUND"


# In[8]: This function contains rules for pronoun and other transformation
def pronoun_transform(token):
    pronoun_dict = dict()
    pronoun_dict["I"] = "YOU"
    pronoun_dict["YOUR"] = "MY"
    pronoun_dict["MY"] = "YOUR"
    pronoun_dict["YOU"] = "I"
    pronoun_dict["ME"] = "YOU"
    pronoun_dict["AM"] = "ARE"
    pronoun_dict["MOTHER"] = "FAMILY"
    pronoun_dict["FATHER"] = "FAMILY"
    pronoun_dict["MOM"] = "FAMILY"
    pronoun_dict["DAD"] = "FAMILY"
    pronoun_dict["WIFE"] = "FAMILY"
    pronoun_dict["BROTHER"] = "FAMILY"
    pronoun_dict["CHILDREN"] = "FAMILY"
    pronoun_dict["SISTER"] = "FAMILY"
    pronoun_dict["FEEL"] = "BELIEF"
    pronoun_dict["THINK"] = "BELIEF"
    pronoun_dict["BELIEVE"] = "BELIEF"
    pronoun_dict["CANT"] = "CAN'T"
    pronoun_dict["WONT"] = "WON'T"
    pronoun_dict["DONT"] = "DON'T"
    pronoun_dict["DREAMED"] = "DREAMT"
    pronoun_dict["DREAMS"] = "DREAMT"

    # Make a list of it
    if pronoun_dict.get(token) is not None:
        return pronoun_dict.get(token)
    return None


# In[9]: Formulating Eliza's greeting, Name tag extraction using POS tagging from NLTK from user input
greetings = ["Hello I am Eliza, what is your name?", "Hey I am Eliza, what is your name?",
             "Hello this is Eliza, what is your name?", "Hello this is Eliza, lets get started with your name."]
print("*" * 60)
print("Welcome!! To exit the application enter 'quit' ")
print("*" * 60)
print("")
print(f"Eliza: {greetings[random.randint(0, len(greetings)) - 1]}\n"
      f"(If you do not wish to give your name just press enter)")
name = str(input("User: "))

stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['hi', 'there', 'whatsup', 'hey', 'hello']
stopwords = [clean_text(word) for word in stopwords]
stopwords.extend(newStopWords)
if name != "":
    name = clean_text(name)
    tokens = nltk.word_tokenize(name)
    tokens_tagged = nltk.pos_tag(tokens)
    tokens_tagged = [token for token in tokens_tagged if token[0].lower() not in set(stopwords)]

    if len(tokens_tagged) != 0:
        for (word, tag) in tokens_tagged:
            if tag in ['NN', 'NNP'] and word != 'Eliza':  # If the word is a proper noun
                name = word
                if tag == 'NNP':
                    break
            elif tag != 'NNP':
                name = "User"
    else:
        name = "User"
else:
    name = "User"

help_text = ["How can I help you today?", "What do you want to talk about?", "How do you feel today?"]
print(f"Eliza: Hi {name}! {help_text[random.randint(0, len(help_text)) - 1]}")

# Loop to get input from user and get replies from Eliza to create a chatbot like conversation.
# It calls the get reply and other funtions described above.
while True:

    try:
        REPLIES = []
        REPLY_LENGTH = 0
        inp = str(input(f"{name}:"))
        if inp.upper() == "QUIT":
            print("thank you for talking to me. If you need me anytime I am just a click away!!\nbye")
            break

        inp = clean_text(inp)
        inp_tok = inp.strip().upper().split()
        key_stack = []
        RANK = 0

        dic_tok_trans = {}
        for token in inp_tok:
            token_rank = get_rank(token)
            if token_rank == "TOKEN_FOUND":
                continue
            elif token_rank == "TOKEN_NOT_FOUND":

                if chatbot_rules['TRANSFORMATION'].get(token) is not None:
                    token1 = chatbot_rules['TRANSFORMATION'].get(token)
                    dic_tok_trans[token1] = token
                    get_rank(token1)

        if len(key_stack) == 0:
            reply = get_reply("NONE", inp)

        for key in key_stack:
            if dic_tok_trans.get(key) is not None:
                inp_text = inp.replace(dic_tok_trans.get(key), key)
            else:
                inp_text = inp

            reply = get_reply(key, inp_text.upper())
            if reply in ["DECOMPOSITION_RULE_NOT_FOUND", "MATCH_NOT_FOUND",
                         "NEXT_KEY"] or reply is None or reply.upper() == 'NONE':
                continue
            else:
                break
            REPLIES.append({key: reply})
    except:
        reply = get_reply("NONE", inp)

    if reply in ["DECOMPOSITION_RULE_NOT_FOUND", "MATCH_NOT_FOUND",
                 "NEXT_KEY"] or reply is None or reply.upper() == 'NONE':
        reply = get_reply("NONE", inp)

    print(f"Eliza: {reply}")
