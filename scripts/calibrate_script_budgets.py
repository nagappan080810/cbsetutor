"""
calibrate_script_budgets.py
─────────────────────────────
Measures REAL chars-per-token ratios for nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
(the model your NIM endpoint is serving, per NVIDIA_MODEL / LLM_PROVIDER=nim) across
scripts, and prints corrected SCRIPT_BUDGET_FACTOR values to paste into concept_graph.py,
replacing the current estimates.

Only downloads the tokenizer (a few MB: tokenizer.json / tokenizer_config.json), NOT the
550B-parameter weights -- AutoTokenizer.from_pretrained fetches tokenizer files only.

Requires network access to huggingface.co (this sandbox's allowlist doesn't include it,
so run this locally / on your own machine, not inside this chat's container).

Usage:
    pip install transformers --break-system-packages
    python calibrate_script_budgets.py
"""
from transformers import AutoTokenizer

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"

# Representative NCERT-style sample sentences per script. Keep these roughly
# equivalent in *meaning* (not exact translations) so token density reflects
# script/tokenizer behavior, not sentence complexity differences.
SAMPLES = {
    "latin": (
        "Ohm's law states that the current flowing through a conductor is directly "
        "proportional to the potential difference across its two ends, provided the "
        "temperature remains constant. This relationship can be written mathematically "
        "as V equals I multiplied by R, where V is voltage, I is current, and R is "
        "resistance measured in ohms."
    ),
    "devanagari": (
        "ओम का नियम बताता है कि किसी चालक से प्रवाहित विद्युत धारा उसके दोनों सिरों के बीच "
        "विभवांतर के समानुपाती होती है, बशर्ते तापमान स्थिर रहे। इस संबंध को गणितीय रूप से "
        "वी बराबर आई गुणा आर के रूप में लिखा जा सकता है, जहाँ वी वोल्टेज है, आई धारा है, और "
        "आर प्रतिरोध है जिसे ओम में मापा जाता है।"
    ),
    "bengali": (
        "ওহমের সূত্র বলে যে একটি পরিবাহীর মধ্য দিয়ে প্রবাহিত তড়িৎ প্রবাহ তার দুই প্রান্তের "
        "মধ্যে বিভব পার্থক্যের সমানুপাতিক, যদি তাপমাত্রা স্থির থাকে। এই সম্পর্কটি গাণিতিকভাবে "
        "লেখা যায় ভি সমান আই গুণ আর, যেখানে ভি ভোল্টেজ, আই প্রবাহ, এবং আর রোধ যা ওহমে পরিমাপ করা হয়।"
    ),
    "tamil": (
        "ஓம் விதி கூறுவது என்னவென்றால், ஒரு கடத்தி வழியாக பாயும் மின்னோட்டம் அதன் இரு "
        "முனைகளுக்கு இடையேயான மின்னழுத்த வேறுபாட்டிற்கு நேர் விகிதாசாரமாக இருக்கும், வெப்பநிலை "
        "மாறாமல் இருந்தால். இந்த தொடர்பை கணிதவியல் ரீதியாக வி சமம் ஐ பெருக்கல் ஆர் என "
        "எழுதலாம், இங்கு வி மின்னழுத்தம், ஐ மின்னோட்டம், மற்றும் ஆர் தடை ஓம் அலகில் அளக்கப்படுகிறது."
    ),
    "telugu": (
        "ఓమ్ నియమం ప్రకారం, ఒక వాహకం గుండా ప్రవహించే విద్యుత్ ప్రవాహం దాని రెండు చివరల మధ్య "
        "ఉన్న విభవాంతరానికి అనులోమానుపాతంలో ఉంటుంది, ఉష్ణోగ్రత స్థిరంగా ఉంటే. ఈ సంబంధాన్ని "
        "గణితశాస్త్రపరంగా వి సమానం ఐ గుణించి ఆర్ గా రాయవచ్చు, ఇక్కడ వి వోల్టేజ్, ఐ ప్రవాహం, "
        "మరియు ఆర్ నిరోధం ఓమ్‌లలో కొలుస్తారు."
    ),
    "kannada": (
        "ಓಮ್‌ನ ನಿಯಮದ ಪ್ರಕಾರ, ಒಂದು ವಾಹಕದ ಮೂಲಕ ಹರಿಯುವ ವಿದ್ಯುತ್ ಪ್ರವಾಹವು ಅದರ ಎರಡು "
        "ತುದಿಗಳ ನಡುವಿನ ವಿಭವಾಂತರಕ್ಕೆ ನೇರ ಅನುಪಾತದಲ್ಲಿರುತ್ತದೆ, ತಾಪಮಾನ ಸ್ಥಿರವಾಗಿದ್ದರೆ. ಈ "
        "ಸಂಬಂಧವನ್ನು ಗಣಿತೀಯವಾಗಿ ವಿ ಸಮ ಐ ಗುಣಿಸಿ ಆರ್ ಎಂದು ಬರೆಯಬಹುದು, ಇಲ್ಲಿ ವಿ ವೋಲ್ಟೇಜ್, "
        "ಐ ಪ್ರವಾಹ, ಮತ್ತು ಆರ್ ಪ್ರತಿರೋಧ ಓಮ್‌ಗಳಲ್ಲಿ ಅಳೆಯಲಾಗುತ್ತದೆ."
    ),
    "malayalam": (
        "ഓം നിയമം പറയുന്നത് ഒരു ചാലകത്തിലൂടെ ഒഴുകുന്ന വൈദ്യുത പ്രവാഹം അതിന്റെ രണ്ട് "
        "അറ്റങ്ങൾക്കിടയിലുള്ള വോൾട്ടേജ് വ്യത്യാസത്തിന് നേർ അനുപാതത്തിലാണ്, താപനില "
        "സ്ഥിരമായിരിക്കുമ്പോൾ. ഈ ബന്ധം ഗണിതശാസ്ത്രപരമായി വി തുല്യം ഐ ഗുണം ആർ എന്ന് "
        "എഴുതാം, ഇവിടെ വി വോൾട്ടേജ്, ഐ പ്രവാഹം, ആർ പ്രതിരോധം ഓമിൽ അളക്കുന്നു."
    ),
    "gujarati": (
        "ઓહ્મનો નિયમ કહે છે કે વાહકમાંથી વહેતો વિદ્યુત પ્રવાહ તેના બે છેડા વચ્ચેના વિભવ "
        "તફાવતના સીધા પ્રમાણમાં હોય છે, જો તાપમાન સ્થિર રહે. આ સંબંધ ગાણિતિક રીતે વી "
        "બરાબર આઈ ગુણ્યા આર તરીકે લખી શકાય છે."
    ),
    "gurmukhi": (
        "ਓਹਮ ਦਾ ਨਿਯਮ ਦੱਸਦਾ ਹੈ ਕਿ ਇੱਕ ਚਾਲਕ ਵਿੱਚੋਂ ਵਹਿਣ ਵਾਲਾ ਬਿਜਲੀ ਦਾ ਕਰੰਟ ਇਸਦੇ ਦੋ "
        "ਸਿਰਿਆਂ ਦੇ ਵਿਚਕਾਰ ਵੋਲਟੇਜ ਦੇ ਅੰਤਰ ਦੇ ਸਿੱਧੇ ਅਨੁਪਾਤ ਵਿੱਚ ਹੁੰਦਾ ਹੈ।"
    ),
    "oriya": (
        "ଓହମ୍ ନିୟମ କହେ ଯେ ଏକ ପରିବାହୀ ମାଧ୍ୟମରେ ପ୍ରବାହିତ ବିଦ୍ୟୁତ ପ୍ରବାହ ଏହାର ଦୁଇ "
        "ମୁଣ୍ଡ ମଧ୍ୟରେ ବିଭବ ପାର୍ଥକ୍ୟ ସହିତ ସିଧାସଳଖ ଅନୁପାତିକ ଅଟେ।"
    ),
}


def main():
    print(f"Loading tokenizer for {MODEL_ID} (tokenizer files only, not model weights)...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    baseline_ratio = None
    results = {}

    for script, text in SAMPLES.items():
        n_tokens = len(tok.encode(text))
        n_chars = len(text)
        chars_per_token = n_chars / n_tokens
        results[script] = chars_per_token
        if script == "latin":
            baseline_ratio = chars_per_token

    print(f"\n{'script':<12} {'chars/token':>12} {'factor vs latin':>18}")
    print("-" * 44)
    factors = {}
    for script, ratio in results.items():
        factor = ratio / baseline_ratio
        factors[script] = round(factor, 2)
        print(f"{script:<12} {ratio:>12.2f} {factor:>18.2f}")

    print("\n--- Paste into concept_graph.py, replacing SCRIPT_BUDGET_FACTOR ---\n")
    print("SCRIPT_BUDGET_FACTOR = {")
    for script, factor in factors.items():
        if script == "latin":
            print(f'    "{script}": 1.0,')
        else:
            # floor at 0.15 -- guards against a pathological ratio producing
            # windows so small the extraction becomes pointless
            print(f'    "{script}": {max(factor, 0.15)},')
    print("}")

    print(
        "\nNote: these ratios come from a handful of representative sentences, not a "
        "full NCERT corpus -- treat them as a solid improvement over the original "
        "estimates, not a final answer. If you see continued truncation errors on a "
        "specific script after applying these, lower that script's factor further."
    )


if __name__ == "__main__":
    main()
