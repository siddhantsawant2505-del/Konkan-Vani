"""
Konkan Vani — Gold Dataset Generator
=====================================
Generates the gold idiom dataset with the ideal schema:
  - konkani_text (Devanagari Konkani)
  - romanized_text (Romi Konkani)
  - marathi_meaning (simple Marathi explanation)
  - english_meaning (English explanation)
  - figurative_meaning (semantic/conceptual meaning)
  - literal_meaning (word-for-word translation)
  - example_sentence (usage in Devanagari)
  - category (thematic grouping)
  - cultural_context (cultural background)
  - source (collection source)

Usage:
    python src/create_gold_dataset.py

Output:
    data/processed/idioms_with_phonetic_keys.csv
"""

import os
import sys
import re
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# ── Phonetic Normalizer ──────────────────────────────────────────────────────
def compute_phonetic_key(text):
    """Normalize Romanized Konkani to a canonical phonetic key."""
    if not isinstance(text, str):
        return ""
    norm = text.lower()
    norm = re.sub(r'aa', 'a', norm)
    norm = re.sub(r'ee', 'i', norm)
    norm = re.sub(r'oo', 'u', norm)
    norm = re.sub(r'ph', 'f', norm)
    norm = re.sub(r'v', 'w', norm)
    norm = re.sub(r'zh', 'j', norm)
    norm = re.sub(r'[^a-z\s]', '', norm)
    norm = " ".join(norm.split())
    return norm

# ── Comprehensive Konkani Idiom Dataset ──────────────────────────────────────
# Each idiom has: konkani_text, romanized_text, marathi_meaning, english_meaning,
# figurative_meaning, literal_meaning, example_sentence, category, cultural_context

IDIOMS_DATA = [
    # ── Character & Integrity ────────────────────────────────────────────────
    {
        "konkani_text": "हाताक चून लावप",
        "romanized_text": "Haatak chun lavap",
        "marathi_meaning": "फसवणे किंवा मूळ हेतू न कळवता नकळत नुकसान करणे",
        "english_meaning": "To deceive someone or to cause them a subtle, unexpected loss",
        "figurative_meaning": "To cheat, swindle, or quietly trick someone while appearing completely harmless",
        "literal_meaning": "To apply limestone (chunam) to someone's hand",
        "example_sentence": "व्यापाऱ्याने गोड बोलून त्याला हाताक चून लावले.",
        "category": "Deception & Fraud",
        "cultural_context": "Derived from the ancient practice of applying edible lime (chuna), which burns or leaves a mark—used metaphorically for cheating or deceiving someone silently.",
    },
    {
        "konkani_text": "हात दाखवून अयाक",
        "romanized_text": "Haat dakhvun ayaak",
        "marathi_meaning": "हात दाखवून येणे म्हणजे खोटे वचन देणे आणि निघून जाणे",
        "english_meaning": "To make false promises and disappear",
        "figurative_meaning": "To lead someone on with false assurances; to make empty promises",
        "literal_meaning": "To show the hand and come (then leave)",
        "example_sentence": "त्याने तुला काम सांगलं आणि त्याने हात दाखवून आयक केलं.",
        "category": "Character & Integrity",
        "cultural_context": "Commonly used in coastal trade markets when someone makes empty assurances about business deals.",
    },
    {
        "konkani_text": "उदक पिऊन विसर",
        "romanized_text": "Udak pionn visor",
        "marathi_meaning": "पाणी पिऊन विसरून जा — म्हणजे गैरसमज विसरून सोडा",
        "english_meaning": "Drink water and forget — let bygones be bygones",
        "figurative_meaning": "To forgive past grievances and move forward without holding grudges",
        "literal_meaning": "Drink water and forget",
        "example_sentence": "जालं ते जालं — आता उदक पिऊन विसर आणि पुढे व्हा.",
        "category": "Peace & Resolution",
        "cultural_context": "An ancient Goan village council sentiment urging neighbors to resolve land disputes amicably.",
    },
    {
        "konkani_text": "मोड्डे मारप",
        "romanized_text": "Modde marap",
        "marathi_meaning": "अचानक अशा गोष्टी सांगणे ज्यामुळे शांतता भंग होते",
        "english_meaning": "To drop a bombshell; to speak out of turn",
        "figurative_meaning": "To make a sudden, tactless, or blunt statement that disrupts social harmony",
        "literal_meaning": "To hit the rice bundle / break boundaries",
        "example_sentence": "सगळी शांतता होती, त्याने अचानक मोड्डे मारून गोंधळ केला.",
        "category": "Speech & Manners",
        "cultural_context": "Derived from harvesting rituals where threshing bundles inappropriately causes grain loss.",
    },
    {
        "konkani_text": "कुळांतलो कोळसो",
        "romanized_text": "Kullantlo kollso",
        "marathi_meaning": "कुटुंबातील अपयशी व्यक्ती — कुटुंबाचे नाव खराब करणारा",
        "english_meaning": "The black sheep of the family",
        "figurative_meaning": "A person who brings disgrace to their respected family",
        "literal_meaning": "Coal from the family clan",
        "example_sentence": "सगळ्यांना सन्मान मिळाला, पण तो एकला कुळांतलो कोळसो झाला.",
        "category": "Character & Integrity",
        "cultural_context": "Refers to coal which blackens everything it touches — a metaphor for family shame.",
    },
    {
        "konkani_text": "ताका पाय ना मुंडो ना",
        "romanized_text": "Taka paay na mudo na",
        "marathi_meaning": "त्याला पायही नाही आणि डोकेही नाही — पूर्णपणे अर्थहीन",
        "english_meaning": "Without rhyme or reason; completely baseless",
        "figurative_meaning": "Something completely illogical, without foundation or structure",
        "literal_meaning": "It has neither legs nor head",
        "example_sentence": "त्याच्या बोलण्यात पाय नाही मुंडो नाही.",
        "category": "Speech & Manners",
        "cultural_context": "Used to dismiss arguments or claims that lack any logical structure.",
    },
    # ── Wisdom & Life Lessons ────────────────────────────────────────────────
    {
        "konkani_text": "दोनूय पायां दोनूय तारवांचेर",
        "romanized_text": "Donui payam donui tarvancheir",
        "marathi_meaning": "दोन्ही पाय दोन्ही वाटांवर — एकाच वेळी दोन विरुद्ध गोष्टी करू नका",
        "english_meaning": "Having a foot in both camps; riding two horses at once",
        "figurative_meaning": "Attempting to balance two opposing commitments, risking disaster",
        "literal_meaning": "One foot on each of two boats",
        "example_sentence": "दोनूय पायां दोनूय तारवांचेर धरून काय फायदा नाही, एका बाजूला ठर.",
        "category": "Wisdom & Life Lessons",
        "cultural_context": "Traditional maritime wisdom from the Mandovi and Zuari estuaries — fishermen know the danger of standing on two boats.",
    },
    {
        "konkani_text": "बांदार बसून उदक पियेवप",
        "romanized_text": "Bandar bosun udak piyeup",
        "marathi_meaning": "बांधावर बसून पाणी पिणे — दुसऱ्याच्या मेहनतीवर मजा करणे",
        "english_meaning": "Reaping where one has not sown; living off others' labor",
        "figurative_meaning": "Living comfortably on the toil and labor of others",
        "literal_meaning": "Sitting on the bund and drinking water",
        "example_sentence": "भावळ्यांना काम न करता, फक्त बांदार बसून उदक पियेवपाची सवय त्याची.",
        "category": "Character & Integrity",
        "cultural_context": "Refers to sitting on an irrigation bund enjoying water while others did the hard work of building it.",
    },
    {
        "konkani_text": "कानाक तेल घालून बसप",
        "romanized_text": "Kanak tel ghalun bosop",
        "marathi_meaning": "कानात तेल टाकून बसणे — ऐकू न येत असे करणे",
        "english_meaning": "Turning a deaf ear; feigning ignorance",
        "figurative_meaning": "Deliberately ignoring warnings, pleas, or advice",
        "literal_meaning": "Pouring oil in one's ears and sitting",
        "example_sentence": "सगळे सांगतात पण तो कानाक तेल घालून बसला.",
        "category": "Speech & Manners",
        "cultural_context": "A vivid image of blocking one's ears — refusal to listen to wise counsel.",
    },
    {
        "konkani_text": "आग्या मोरांक पांखां",
        "romanized_text": "Aagya morank pankham",
        "marathi_meaning": "आवळ्यातील मोर पिसारा फुलवतात — दिसावटीसाठी अभिमान करणे",
        "english_meaning": "Empty vessels make the most noise; showing off without substance",
        "figurative_meaning": "Superficial pride without real ability or achievement",
        "literal_meaning": "Peacocks in the yard showing off feathers",
        "example_sentence": "त्याला काहीच माहित नाही, पण आग्या मोरांक पांखां करतो.",
        "category": "Pride & Humility",
        "cultural_context": "Reflects the rich fauna of the Western Ghats and agricultural parables of Goa.",
    },
    {
        "konkani_text": "मांजराक दूध राखपाक दिवप",
        "romanized_text": "Mazrak doodh rakhpak divop",
        "marathi_meaning": "मांजराला दूध साठवून ठेवायला सांगणे — अनिश्चित व्यक्तीला विश्वास ठेवणे",
        "english_meaning": "Putting the fox in charge of the hen house",
        "figurative_meaning": "Entrusting something valuable to an untrustworthy person",
        "literal_meaning": "Giving a cat milk to guard",
        "example_sentence": "त्याच्या कोठ्यात तिजोरी दिली तर मांजराक दूध राखपाक दिवप ठरेल.",
        "category": "Wisdom & Life Lessons",
        "cultural_context": "Universal metaphor adapted to Konkani household life where cats guarding milk is a familiar image.",
    },
    {
        "konkani_text": "कोंब्याक पांखां फुटप",
        "romanized_text": "Kombyak pankham futap",
        "marathi_meaning": "कोंबऱ्याला पिसारा पंख फुटले — नवीन अधिकार मिळाल्यावर अभिमान करणे",
        "english_meaning": "Getting too big for one's boots",
        "figurative_meaning": "A subordinate acting beyond their station with newfound arrogance",
        "literal_meaning": "Rooster growing flying wings",
        "example_sentence": "नवीन नोकरी मिळतच त्याला कोंब्याक पांखां फुटल्या कोशी दिसतं.",
        "category": "Pride & Humility",
        "cultural_context": "A rooster cannot fly far — metaphor for someone who gains a little power and becomes arrogant.",
    },
    {
        "konkani_text": "घराचेर कौलां नात",
        "romanized_text": "Ghoracher koulam nant",
        "marathi_meaning": "घराला मेट नाही — आपल्या कुटुंबातील कमतरता जगाला दाखवणे",
        "english_meaning": "People in glass houses shouldn't throw stones",
        "figurative_meaning": "Exposing one's internal family weaknesses to the outside world",
        "literal_meaning": "No roof tiles on the house",
        "example_sentence": "आपल्या घराचेर कौलां नाहीत असताना दुसऱ्याच्या फटक्या मारू नका.",
        "category": "Wisdom & Life Lessons",
        "cultural_context": "Roof tiles (koulam) protect the household — without them, everything is exposed.",
    },
    {
        "konkani_text": "एक हात पोट्यार",
        "romanized_text": "Ek haat potyaar",
        "marathi_meaning": "एक हात पोटावर — अत्यंत गरिबीत जगणे",
        "english_meaning": "Living hand to mouth",
        "figurative_meaning": "Living in severe poverty or extreme frugality",
        "literal_meaning": "One hand on the belly",
        "example_sentence": "त्याच्या घरी एक हात पोट्यार असताना देखून त्यांनी चिंता केली.",
        "category": "Livelihood & Sea",
        "cultural_context": "Common among coastal fisherfolk describing seasons when monsoon bans halt fishing.",
    },
    {
        "konkani_text": "नून जिल्यार मास",
        "romanized_text": "Noon zilyar maas",
        "marathi_meaning": "मीठ टाकलेला मासा — योग्य प्रयत्न केल्यावरच गोडी येते",
        "english_meaning": "Patience and care bring true worth",
        "figurative_meaning": "Things only gain value and appreciation with proper effort and care",
        "literal_meaning": "Meat when seasoned with salt",
        "example_sentence": "कष्ट केल्यावरच नून जिल्यार मासा म्हणजे फळ मिळतं.",
        "category": "Wisdom & Life Lessons",
        "cultural_context": "From Goan culinary tradition — properly seasoned food requires patience.",
    },
    {
        "konkani_text": "पाण्याच्या विहिरीत खळखळ पाडप",
        "romanized_text": "Panyachi vihirrit khalkhal padap",
        "marathi_meaning": "पाण्याच्या विहिरीत खळखळ — अशा गोष्टीत गोंधळ निर्माण करणे जी शांत आहेत",
        "english_meaning": "Making waves in still water; stirring up trouble unnecessarily",
        "figurative_meaning": "Creating disturbance in a peaceful situation for no reason",
        "literal_meaning": "Creating ripples in a well",
        "example_sentence": "ती शांत आहे तिला पाण्याच्या विहिरीत खळखळ पाडू नका.",
        "category": "Speech & Manners",
        "cultural_context": "Wells (vihir) were communal water sources — disturbing the water affected everyone.",
    },
    {
        "konkani_text": "तेल घालून पोळी भाजप",
        "romanized_text": "Tel ghalun poli bhajap",
        "marathi_meaning": "तेल टाकून पोळी भाजणे — मोठ्या तयारीसाठी साधने लागतात",
        "english_meaning": "You need the right resources for the job",
        "figurative_meaning": "Preparation and proper resources are essential before attempting something",
        "literal_meaning": "Pour oil and roast the flatbread",
        "example_sentence": "तेल घालून पोळी भाजायला हवी, फक्त हाताने भाजता येत नाही.",
        "category": "Livelihood & Sea",
        "cultural_context": "From kitchen wisdom — good cooking requires proper ingredients and technique.",
    },
    {
        "konkani_text": "म्हैन्याच्या पाठीशी लागप",
        "romanized_text": "Mhainyachi paatshi laap",
        "marathi_meaning": "म्हैनीच्या पाठीशी लागणे — अत्यंत जवळ राहून त्रास देणे",
        "english_meaning": "To be a constant nuisance; to stick like a leech",
        "figurative_meaning": "Being persistently annoying or clingy to someone",
        "literal_meaning": "To cling to the leech's back",
        "example_sentence": "तो म्हैन्याच्या पाठीशी लागला आहे, सोडत नाही.",
        "category": "Speech & Manners",
        "cultural_context": "Leeches (mhaini) are common in Goan monsoon fields — a vivid image of persistent annoyance.",
    },
    {
        "konkani_text": "शेवटाला शिंक मारप",
        "romanized_text": "Shevtala shink maarap",
        "marathi_meaning": "शेवटी शिंक मारणे — अखेर अपयश भोगणे",
        "english_meaning": "To get stung at the end; to face consequences eventually",
        "figurative_meaning": "Eventually facing the negative consequences of one's actions",
        "literal_meaning": "The scorpion stings at the end",
        "example_sentence": "खोटं बोलणाऱ्याला शेवटाला शिंक मारते.",
        "category": "Wisdom & Life Lessons",
        "cultural_context": "Scorpions (shink) are common in rural Goa — a warning that harmful actions have delayed consequences.",
    },
    {
        "konkani_text": "आळ्याच्या झुडूपात उंबरा फोडप",
        "romanized_text": "Aalyachya zhudupaat umbra fodap",
        "marathi_meaning": "आळ्याच्या झुडूपात उंबरा फोडणे — अनावश्यक गोंधळ करणे",
        "english_meaning": "To make a mountain out of a molehill",
        "figurative_meaning": "Creating unnecessary trouble or exaggerating small issues",
        "literal_meaning": "To dig a pit in an anthill",
        "example_sentence": "साधी गोष्ट आहे, आळ्याच्या झुडूपात उंबरा फोडू नका.",
        "category": "Speech & Manners",
        "cultural_context": "Ant hills (aalyache zhudup) are fragile structures — disturbing them creates pointless mess.",
    },
    {
        "konkani_text": "कोलंबीत कोंबा मारप",
        "romanized_text": "Kolambit komba maarap",
        "marathi_meaning": "कोलंबीत कोंबा मारणे — स्वतःच्या भोवती गोंधळ करणे",
        "english_meaning": "To stir up trouble in one's own neighborhood",
        "figurative_meaning": "Causing problems in one's own community or family",
        "literal_meaning": "To beat a drum in the pigeon coop",
        "example_sentence": "तो कोलंबीत कोंबा मारतो, म्हणजे आपल्याच घरात गोंधळ करतो.",
        "category": "Character & Integrity",
        "cultural_context": "Pigeon coops (kolambi) are quiet — drumming there causes chaos among the birds.",
    },
    {
        "konkani_text": "खरे बोलप म्हणजे भांडप",
        "romanized_text": "Khare bolap mhanje bhandap",
        "marathi_meaning": "खरं बोलणे म्हणजे भांडणे — सत्य बोलल्यावर वाद होतो",
        "english_meaning": "Truth causes conflict; honesty is often unwelcome",
        "figurative_meaning": "Speaking the truth often leads to arguments and confrontation",
        "literal_meaning": "To speak truthfully means to quarrel",
        "example_sentence": "खरे बोलप म्हणजे भांडप, पण खरं तेच बरोबर आहे.",
        "category": "Speech & Manners",
        "cultural_context": "A pragmatic observation about social dynamics in close-knit Goan communities.",
    },
    {
        "konkani_text": "सापडलेल्याला सांगप",
        "romanized_text": "Sapadlyala sangap",
        "marathi_meaning": "सापडलेल्याला सांगणे — ज्याला माहीत आहे त्याला सांगू नका",
        "english_meaning": "Teaching your grandmother to suck eggs",
        "figurative_meaning": "Trying to teach something to someone who already knows it better",
        "literal_meaning": "To tell someone who has already found it",
        "example_sentence": "त्याला माहीत आहे, सापडलेल्याला सांगू नका.",
        "category": "Speech & Manners",
        "cultural_context": "A common caution against unsolicited advice to experts.",
    },
    {
        "konkani_text": "गाय दुधारी असता",
        "romanized_text": "Gaay dudhari asta",
        "marathi_meaning": "गाय दुधारी असते — प्रत्येकाला काही न काही गुण असतो",
        "english_meaning": "Every cow gives milk; everyone has some talent",
        "figurative_meaning": "Everyone has some positive quality or skill, even if it's not obvious",
        "literal_meaning": "The cow is milk-bearing",
        "example_sentence": "गाय दुधारी असते, प्रत्येकाकडे काही न काही चांगलं असतं.",
        "category": "Pride & Humility",
        "cultural_context": "A positive proverb encouraging people to recognize value in everyone.",
    },
    {
        "konkani_text": "पाण्यावर लिहिलेले",
        "romanized_text": "Panyavar lihilele",
        "marathi_meaning": "पाण्यावर लिहिलेले — कायमस्वरूपी नसलेले, तात्पुरते",
        "english_meaning": "Written on water; temporary and impermanent",
        "figurative_meaning": "Something that won't last or has no lasting impact",
        "literal_meaning": "Written on water",
        "example_sentence": "त्याच्या वचनांवर विश्वास ठेवू नका, ते पाण्यावर लिहिलेले आहेत.",
        "category": "Wisdom & Life Lessons",
        "cultural_context": "A universal metaphor — writing on water leaves no trace.",
    },
    {
        "konkani_text": "डोळ्यांत धूळ घालचें",
        "romanized_text": "Dollyant dhool ghalchem",
        "marathi_meaning": "डोळ्यांत धूळ टाकणे — कुणाला फसवणे वा भ्रमित करणे",
        "english_meaning": "To pull the wool over someone's eyes",
        "figurative_meaning": "To deceive or mislead someone intentionally",
        "literal_meaning": "To throw dust into the eyes",
        "example_sentence": "त्याने सगळ्यांच्या डोळ्यांत धूळ घालून पैसे चोरले.",
        "category": "Character & Integrity",
        "cultural_context": "A vivid image of blinding someone with dust to steal — metaphor for deception.",
    },
    {
        "konkani_text": "कळी उगवप म्हणजे फुल फुटप",
        "romanized_text": "Kali ugavap mhanje phul futap",
        "marathi_meaning": "कळी उगवणे म्हणजे फूल फुटणे — वेळ आल्यावर गोडी येते",
        "english_meaning": "Every bud blooms in time; patience brings results",
        "figurative_meaning": "Good things come to those who wait; everyone has their moment",
        "literal_meaning": "When the bud sprouts, the flower blooms",
        "example_sentence": "कळी उगवप म्हणजे फुल फुटप, तुझा वेळ आला तर येईल.",
        "category": "Wisdom & Life Lessons",
        "cultural_context": "From the rich botanical imagery of Goa's tropical flora.",
    },
    {
        "konkani_text": "हिरव्यागार झालेला",
        "romanized_text": "Hirvyaagar zhaalela",
        "marathi_meaning": "हिरव्यागार झालेला — अत्यंत स्वस्त व तरुण झालेला",
        "english_meaning": "Turned green; to become very healthy or prosperous",
        "figurative_meaning": "Someone who has flourished or prospered greatly",
        "literal_meaning": "Having turned green",
        "example_sentence": "नोकरी मिळाल्यापासून तो हिरव्यागार झाला आहे.",
        "category": "Livelihood & Sea",
        "cultural_context": "Green represents prosperity and health in Konkani culture — lush vegetation means good harvest.",
    },
    {
        "konkani_text": "पोळीत पाणी गेलें",
        "romanized_text": "Polit paani gele",
        "marathi_meaning": "पोळीत पाणी गेले — सर्व प्रयत्न व्यर्थ झाले",
        "english_meaning": "All effort went to waste; the plan fell through",
        "figurative_meaning": "When hard work produces no results despite best efforts",
        "literal_meaning": "Water got into the flatbread (making it soggy)",
        "example_sentence": "त्याने इतकी मेहनत केली पण पोळीत पाणी गेलं.",
        "category": "Livelihood & Sea",
        "cultural_context": "From kitchen life — soggy poli is ruined, just like wasted effort.",
    },
    {
        "konkani_text": "मातीत मिठा गालप",
        "romanized_text": "Maatit mitha gaalap",
        "marathi_meaning": "मातीत मीठ गाळणे — अनावश्यक गोष्टीत पैसे खर्च करणे",
        "english_meaning": "Wasting resources on something futile",
        "figurative_meaning": "Investing time or money in something that won't yield returns",
        "literal_meaning": "Straining salt into the soil",
        "example_sentence": "त्या प्रकल्पात मातीत मिठा गाळू नका.",
        "category": "Wisdom & Life Lessons",
        "cultural_context": "Salt dissolving in soil is lost forever — a warning against wasteful spending.",
    },
    {
        "konkani_text": "न्हायल्या मांजरासारखा",
        "romanized_text": "Nhaylya mazrasarkha",
        "marathi_meaning": "न्हायल्या मांजरासारखा — स्वच्छ व सुंदर दिसणे",
        "english_meaning": "Clean as a washed cat; looking fresh and neat",
        "figurative_meaning": "Looking freshly groomed and presentable",
        "literal_meaning": "Like a cat that has been bathed",
        "example_sentence": "ती न्हायल्या मांजरासारखी सुंदर दिसते.",
        "category": "Speech & Manners",
        "cultural_context": "Cats groom themselves meticulously — a washed cat looks particularly neat.",
    },
    {
        "konkani_text": "कुंभाराच्या भांड्यासारखे",
        "romanized_text": "Kumbharyachya bhandyasarkhe",
        "marathi_meaning": "कुंभाराच्या भांड्यासारखे — एकसारखे व नीट असणे",
        "english_meaning": "All alike as potter's vessels; uniform and orderly",
        "figurative_meaning": "Everything or everyone following the same pattern without individuality",
        "literal_meaning": "Like the potter's vessels",
        "example_sentence": "हे सगळे कुंभाराच्या भांड्यासारखे एकसारखे आहेत.",
        "category": "Speech & Manners",
        "cultural_context": "Potters make similar vessels from the same clay — a metaphor for conformity.",
    },
    {
        "konkani_text": "भाताच्या पोळीत तूप",
        "romanized_text": "Bhatyachi polit toop",
        "marathi_meaning": "भाताच्या पोळीत तूप — कष्टानंतर योग्य बक्षीस मिळणे",
        "english_meaning": "The reward after hard work; the icing on the cake",
        "figurative_meaning": "Receiving the deserved reward after sustained effort",
        "literal_meaning": "Ghee on the rice flatbread",
        "example_sentence": "त्याने इतकी मेहनत केली, भाताच्या पोळीत तूप झालं.",
        "category": "Livelihood & Sea",
        "cultural_context": "Rice poli with ghee is a celebratory Goan meal — represents the sweet reward of effort.",
    },
    {
        "konkani_text": "वाघोळीत वाघ सापडप",
        "romanized_text": "Vagholit vaagh sapdap",
        "marathi_meaning": "वाघोळीत वाघ सापडणे — जिथे अपाय असतो तिथेच जाणे",
        "english_meaning": "Walking into the tiger's den; walking into danger knowingly",
        "figurative_meaning": "Deliberately putting oneself in a dangerous or difficult situation",
        "literal_meaning": "The tiger appears in the tiger's territory",
        "example_sentence": "तो वाघोळीत वाघ सापडल्यासारखा गेला.",
        "category": "Wisdom & Life Lessons",
        "cultural_context": "Tigers (vaagh) were historically found in Goa's Western Ghats forests.",
    },
    {
        "konkani_text": "कोंदळीत कोंदळ घालप",
        "romanized_text": "Kondalit kondal ghaalap",
        "marathi_meaning": "कोंदळीत कोंदळ टाकणे — अशा गोष्टीत अडचण निर्माण करणे जी आधीच अडचणीच्या आहेत",
        "english_meaning": "Adding fuel to the fire; making a bad situation worse",
        "figurative_meaning": "Intensifying an already difficult situation",
        "literal_meaning": "Adding embers to the embers",
        "example_sentence": "त्याने कोंदळीत कोंदळ घालून गोंधळ वाढवला.",
        "category": "Speech & Manners",
        "cultural_context": "From traditional wood-fire cooking — adding more embers to a dying fire disrupts the cooking.",
    },
    {
        "konkani_text": "माणसाला माणूस बनवप",
        "romanized_text": "Mansala manus banavap",
        "marathi_meaning": "माणसाला माणूस बनवणे — चांगला व्यक्तिमत्त्व घडवणे",
        "english_meaning": "To make a man out of someone; to build character",
        "figurative_meaning": "Developing someone's character through guidance and experience",
        "literal_meaning": "To make a human into a proper person",
        "example_sentence": "शिक्षणाने माणसाला माणूस बनवते.",
        "category": "Character & Integrity",
        "cultural_context": "Emphasizes that education and upbringing shape true character.",
    },
    {
        "konkani_text": "सागरी लांब जायला हवे",
        "romanized_text": "Saagari laamb jaayla have",
        "marathi_meaning": "सागरी लांब जावे लागते — मोठ्या गोष्टींसाठी धैर्य हवे",
        "english_meaning": "One must go far into the sea; great rewards require great risks",
        "figurative_meaning": "To achieve something significant, one must venture beyond comfort zones",
        "literal_meaning": "One must go far into the ocean",
        "example_sentence": "सागरी लांब जायला हवे, मगच मोठी मासे सापडतील.",
        "category": "Livelihood & Sea",
        "cultural_context": "Konkani fishermen know that the best catch comes from venturing far into the Arabian Sea.",
    },
    {
        "konkani_text": "झिंगा मारून भाजप",
        "romanized_text": "Jhinga maarun bhajap",
        "marathi_meaning": "झिंगा मारून भाजणे — उत्तम साहित्य तयार करण्यासाठी मेहनत लागते",
        "english_meaning": "To prepare something special with care and effort",
        "figurative_meaning": "Creating something valuable requires dedicated effort",
        "literal_meaning": "To fry prawns after catching them",
        "example_sentence": "झिंगा मारून भाजायला हवे, फक्त पकडणं पुरत नाही.",
        "category": "Livelihood & Sea",
        "cultural_context": "Prawns (jhinga) are a prized Goan delicacy — preparing them well requires skill.",
    },
    {
        "konkani_text": "सागरी पाऊस आला",
        "romanized_text": "Saagari paus aala",
        "marathi_meaning": "सागरी पाऊस आला — मोठी आनंदाची बाब घडली",
        "english_meaning": "Rain in the sea; a moment of great joy or relief",
        "figurative_meaning": "A long-awaited positive event that brings relief and happiness",
        "literal_meaning": "Rain has come to the sea",
        "example_sentence": "त्याचा नातू जन्माला आला — सागरी पाऊस आला!",
        "category": "Livelihood & Sea",
        "cultural_context": "The monsoon arriving at sea signals the end of the fishing ban and renewal of livelihood.",
    },
    {
        "konkani_text": "कापूस धुणप",
        "romanized_text": "Kaapus dhunap",
        "marathi_meaning": "कापूस धुणणे — अनावश्यक मेहनत करणे",
        "english_meaning": "Beating cotton; doing pointless work",
        "figurative_meaning": "Engaging in labor that produces no meaningful result",
        "literal_meaning": "To beat/clean cotton",
        "example_sentence": "हे कापूस धुणप सोयीस नाही, काही उपयोगी काम कर.",
        "category": "Livelihood & Sea",
        "cultural_context": "Cotton processing was traditional work — beating already-clean cotton wastes effort.",
    },
    {
        "konkani_text": "गावाकडच्या देवळाला गाठप",
        "romanized_text": "Gaavakadchya devlala gaathap",
        "marathi_meaning": "गावाकडच्या देवळाला गाठणे — जवळच्या गोष्टींची कदर न करणे",
        "english_meaning": "Not recognizing the temple in your own village; taking local blessings for granted",
        "figurative_meaning": "Failing to appreciate what is close and familiar",
        "literal_meaning": "To find the temple near your own village",
        "example_sentence": "गावाकडच्या देवळाला गाठ, बाहेर कशाला जातोस?",
        "category": "Wisdom & Life Lessons",
        "cultural_context": "Encourages valuing local resources and relationships over distant ones.",
    },
    {
        "konkani_text": "कुशीत कुठली ठेवप",
        "romanized_text": "Kushit kuthli thevap",
        "marathi_meaning": "कुशीत कुठली ठेवणे — गुप्त ठेवणे, राखून ठेवणे",
        "english_meaning": "To keep something hidden; to hold a secret close",
        "figurative_meaning": "Keeping valuable knowledge or resources hidden for later use",
        "literal_meaning": "To place in the folds of the garment",
        "example_sentence": "त्याने कुशीत कुठली ठेवली, कोणाला सांगितली नाही.",
        "category": "Character & Integrity",
        "cultural_context": "Traditional Konkani garments had folds used to carry valuables — keeping things close to the body.",
    },
    {
        "konkani_text": "मातीत माती करप",
        "romanized_text": "Maatit maati karap",
        "marathi_meaning": "मातीत माती करणे — अनावश्यक मेहनत करणे",
        "english_meaning": "Mixing soil with soil; doing redundant work",
        "figurative_meaning": "Performing unnecessary tasks that don't add value",
        "literal_meaning": "To mix soil into soil",
        "example_sentence": "तू मातीत माती करतोस, यात काही फायदा नाही.",
        "category": "Livelihood & Sea",
        "cultural_context": "Farmers know that mixing good soil with more soil doesn't improve it.",
    },
    {
        "konkani_text": "बेतालाच्या बेगीत बेत",
        "romanized_text": "Betalachya begit bet",
        "marathi_meaning": "बेतालाच्या बेगीत बेत — अनावश्यक गोष्टींत पैसे वाया करणे",
        "english_meaning": "Wasting money on unnecessary things",
        "figurative_meaning": "Spending resources on things that aren't essential",
        "literal_meaning": "Music in the musician's bag (unheard)",
        "example_sentence": "बेतालाच्या बेगीत बेत ठेवू नका, उपयोगी गोष्टींवर खर्च कर.",
        "category": "Livelihood & Sea",
        "cultural_context": "Musicians carry instruments in bags — music unheard is potential wasted.",
    },
    {
        "konkani_text": "तळ्याच्या पाण्यात मासे शिंपडप",
        "romanized_text": "Talyachya paanyat maase shimpdap",
        "marathi_meaning": "तळ्याच्या पाण्यात मासे शिंपडणे — स्वतःच्या गावात अपयश भोगणे",
        "english_meaning": "Fish dying in the pond; facing failure in one's own domain",
        "figurative_meaning": "Failing in one's own area of expertise or comfort zone",
        "literal_meaning": "Fish floating dead in pond water",
        "example_sentence": "तळ्याच्या पाण्यात मासे शिंपडले — त्याच्या गावातच त्याला अपयश आलं.",
        "category": "Livelihood & Sea",
        "cultural_context": "A fish dying in a pond suggests something fundamentally wrong — failure where success should be easy.",
    },
    {
        "konkani_text": "साधू बनून भोंदू भाजप",
        "romanized_text": "Saadhu banun bhoodu bhajap",
        "marathi_meaning": "साधू बनून भोंदू भाजणे — दिसावटीसाठी साधू बनणे पण आत वेगळं असणे",
        "english_meaning": "To pretend to be holy while being corrupt inside",
        "figurative_meaning": "Hypocrisy — appearing righteous while acting immorally",
        "literal_meaning": "Becoming a saint to roast (earn) bread",
        "example_sentence": "साधू बनून भोंदू भाजतात, त्यांना विश्वास ठेवू नका.",
        "category": "Character & Integrity",
        "cultural_context": "A warning against religious hypocrisy — those who perform piety for material gain.",
    },
    {
        "konkani_text": "उन्हाळ्यात पाऊस",
        "romanized_text": "Unhaalyat paus",
        "marathi_meaning": "उन्हाळ्यात पाऊस — अप्रत्याशित आनंदाची बाब",
        "english_meaning": "Rain in summer; an unexpected pleasant surprise",
        "figurative_meaning": "Something wonderful happening when least expected",
        "literal_meaning": "Rain in the summer",
        "example_sentence": "त्याचा अचानक येणे उन्हाळ्यात पाऊस झाला!",
        "category": "Peace & Resolution",
        "cultural_context": "Summer rain in tropical Goa is a rare blessing — a metaphor for unexpected joy.",
    },
    {
        "konkani_text": "फुलांना मळ लागप",
        "romanized_text": "Phulanna mal laap",
        "marathi_meaning": "फुलांना मळ लागणे — सुंदर गोष्टींना नुकसान होणे",
        "english_meaning": "Dust on flowers; beauty being tarnished",
        "figurative_meaning": "Something beautiful or pure being spoiled by negative influences",
        "literal_meaning": "Flowers getting covered in dust",
        "example_sentence": "तिच्या चेहऱ्यावरचा आनंद फुलांना मळ लागल्यासारखा झाला.",
        "category": "Peace & Resolution",
        "cultural_context": "Flowers in Goan gardens are cherished — dust spoiling them represents loss of beauty or joy.",
    },
    {
        "konkani_text": "कुलूप लावप म्हणजे भांडप",
        "romanized_text": "Kulup laavap mhanje bhandap",
        "marathi_meaning": "कुलूप लावणे म्हणजे भांडणे — शेजारी असताना वाद होतो",
        "english_meaning": "Locking the door leads to quarrels; closeness breeds conflict",
        "figurative_meaning": "Living in close proximity naturally leads to disagreements",
        "literal_meaning": "Putting on the lock means quarreling",
        "example_sentence": "कुलूप लावप म्हणजे भांडप, शेजारी राहण्यात अडचण येते.",
        "category": "Speech & Manners",
        "cultural_context": "Shared walls in Goan houses mean neighbors hear everything — privacy leads to tension.",
    },
    {
        "konkani_text": "कोंबऱ्याला कोंब घालप",
        "romanized_text": "Kombyala komb ghaalap",
        "marathi_meaning": "कोंबऱ्याला कोंब घालणे — अनावश्यक गोष्ट देणे",
        "english_meaning": "Giving a rooster a comb; adding something unnecessary to someone who already has it",
        "figurative_meaning": "Providing something redundant to someone who already possesses it",
        "literal_meaning": "To put a comb on a rooster",
        "example_sentence": "त्याला आधीच आहे, कोंबऱ्याला कोंब घालू नका.",
        "category": "Speech & Manners",
        "cultural_context": "A rooster already has a comb — adding another is pointless redundancy.",
    },
    {
        "konkani_text": "ताल्याच्या पाण्यात माती",
        "romanized_text": "Talyachya paanyat maati",
        "marathi_meaning": "ताल्याच्या पाण्यात माती — शांत गोष्टीत गोंधळ निर्माण करणे",
        "english_meaning": "Mud in the lake; contaminating something pure",
        "figurative_meaning": "Spoiling something clean or peaceful with unnecessary interference",
        "literal_meaning": "Soil in the lake water",
        "example_sentence": "ताल्याच्या पाण्यात माती करू नका, शांत रहा.",
        "category": "Peace & Resolution",
        "cultural_context": "Lakes (taalya) are vital water sources — contaminating them affects the whole community.",
    },
    {
        "konkani_text": "सोनेरी माशी",
        "romanized_text": "Soneri maashi",
        "marathi_meaning": "सोनेरी माशी — अत्यंत मौल्यवान गोष्ट",
        "english_meaning": "A golden fish; something extremely valuable or rare",
        "figurative_meaning": "An exceptionally valuable person or opportunity",
        "literal_meaning": "Golden fish",
        "example_sentence": "ती सोनेरी माशी आहे, तिची कदर कर.",
        "category": "Pride & Humility",
        "cultural_context": "Golden fish are rare treasures in Goan rivers — a metaphor for precious people.",
    },
    {
        "konkani_text": "विंदीच्या ज्योतीत तूप",
        "romanized_text": "Vindichya jyotit toop",
        "marathi_meaning": "विंदीच्या ज्योतीत तूप — अल्प साधनांत मोठे कार्य करणे",
        "english_meaning": "Ghee on a small flame; making the most of limited resources",
        "figurative_meaning": "Achieving great things with minimal resources through wisdom",
        "literal_meaning": "Ghee in the oil lamp's flame",
        "example_sentence": "विंदीच्या ज्योतीत तूप टाकून त्यांनी उजाळा केला.",
        "category": "Wisdom & Life Lessons",
        "cultural_context": "A small oil lamp (vindi) with a drop of ghee illuminates a whole room — minimal effort, maximum impact.",
    },
    {
        "konkani_text": "सागरात उब मापप",
        "romanized_text": "Saagarat ub maapap",
        "marathi_meaning": "सागरात उब मापणे — अशक्य गोष्टी करायचा प्रयत्न करणे",
        "english_meaning": "Measuring the warmth of the ocean; attempting the impossible",
        "figurative_meaning": "Trying to do something that cannot be measured or accomplished",
        "literal_meaning": "To measure the warmth in the ocean",
        "example_sentence": "सागरात उब मापता येत नाही, तू अशक्य गोष्ट करतोस.",
        "category": "Wisdom & Life Lessons",
        "cultural_context": "The ocean's warmth is immeasurable — a metaphor for impossible tasks.",
    },
]

# ── Generate CSV ─────────────────────────────────────────────────────────────
def main():
    print("=== KONKAN VANI — GOLD DATASET GENERATOR ===")
    print(f"Generating {len(IDIOMS_DATA)} idiom entries...")
    
    # Add script and phonetic keys
    for item in IDIOMS_DATA:
        item["script"] = "Devanagari"
        item["phonetic_key"] = compute_phonetic_key(item["romanized_text"])
    
    df = pd.DataFrame(IDIOMS_DATA)
    
    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)
    
    output_path = "data/processed/idioms_with_phonetic_keys.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"\n✓ Saved {output_path}")
    print(f"  Total entries: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Category breakdown
    print("\n--- Category Breakdown ---")
    for cat, count in df["category"].value_counts().items():
        print(f"  {cat}: {count}")
    
    # Script breakdown
    print("\n--- Script Breakdown ---")
    for sc, count in df["script"].value_counts().items():
        print(f"  {sc}: {count}")
    
    print("\n✓ Gold dataset generation complete!")
    print(f"  Next step: Run server-nlp/scripts/build_embeddings.py to generate embeddings")
    
    print("\n✓ Gold dataset generation complete!")
    print(f"  Next step: Run server-nlp/scripts/build_embeddings.py to generate embeddings")

if __name__ == "__main__":
    main()
