"""Held-out Bible reasoning probes, authored independently of the domain trainer.

Do not include this file in any Kaggle training notebook. It is a small
research check, not a definitive theological benchmark.
"""
from __future__ import annotations
import json
from pathlib import Path
from bible_embeddings.corpus import load_passages
from training.build_domain_data import overlapping_refs

# One direct entailment, one explicit contradiction, one topical but unproven
# modern claim per passage. None of these exact passages are in training.
PROBES = [
 ("Genesis 28:12", "Jacob dreamed of a ladder reaching heaven with angels on it.",
  "Jacob had no dream about a ladder or angels.", "Jacob used an electric escalator to reach heaven."),
 ("Genesis 50:20", "Joseph says his brothers intended evil, but God intended good.",
  "Joseph says his brothers intended good while God intended evil.", "The brothers later opened a commercial shipping business."),
 ("Exodus 18:21", "Moses is told to select capable and truthful people as rulers.",
  "Moses is ordered to select corrupt and dishonest rulers.", "The rulers are chosen by computer algorithm."),
 ("Deuteronomy 4:2", "The people are told neither to add to nor subtract from the commandments.",
  "The people are instructed to add new commandments to God's words.", "Modern software should prohibit editing all documents."),
 ("Joshua 6:20", "The walls of Jericho fell after the people shouted and trumpets sounded.",
  "The walls of Jericho did not fall when the people shouted.", "The walls were demolished using radio-controlled explosives."),
 ("Judges 7:7", "God says Gideon's 300 men will be used to defeat the Midianites.",
  "God says Gideon's 300 men cannot be used to save them.", "Gideon's troops were equipped with drones."),
 ("1 Samuel 17:49", "David struck the Philistine's forehead with a stone.",
  "David missed the Philistine completely with his stone.", "David's stone was radioactive."),
 ("2 Samuel 11:27", "The thing David had done displeased the LORD.",
  "The LORD was pleased with everything David had just done.", "David forged an electronic signature."),
 ("1 Kings 18:38", "The LORD's fire consumed the sacrifice.",
  "The sacrifice was explicitly untouched by the LORD's fire.", "A gas-powered lighter started the fire."),
 ("2 Kings 2:11", "Elijah went to heaven by a whirlwind.",
  "Elijah was not taken up in a whirlwind.", "Elijah traveled in a combustion-engine vehicle."),
 ("Psalms 19:1", "The heavens declare God's glory, according to the psalm.",
  "The heavens are said to conceal all of God's glory.", "This verse confirms life on another planet."),
 ("Ecclesiastes 12:13", "The writer advises fearing God and keeping his commandments.",
  "The writer advises ignoring God's commandments.", "Every modern government must require church membership."),
 ("Isaiah 55:8", "The LORD says his thoughts are different from human thoughts.",
  "The LORD says his thoughts are identical to human thoughts.", "Modern psychology is forbidden by this verse."),
 ("Jeremiah 29:11", "The LORD describes his intentions toward the people as peaceful.",
  "The LORD describes only evil intentions toward the people.", "Every investment will increase in value tomorrow."),
 ("Lamentations 3:22", "The LORD's compassion is said not to fail.",
  "The LORD's compassion is said to have failed completely.", "Antibiotics had already been invented."),
 ("Luke 15:20", "The father ran to meet and embrace his returning son.",
  "The father refused to meet or embrace his returning son.", "The father owned a modern electric car."),
 ("John 11:35", "Jesus wept.",
  "Jesus did not weep.", "Jesus was crying because of a pollen allergy."),
 ("Acts 20:35", "Jesus is quoted as saying that giving is more blessed than receiving.",
  "Jesus is quoted as saying that receiving is more blessed than giving.", "Every charitable contribution guarantees personal wealth."),
 ("Romans 6:23", "The wages of sin are described as death.",
  "The wages of sin are described as eternal life.", "Civil law should execute a person for every sin."),
 ("1 Corinthians 15:14", "Paul says preaching would be vain if Christ had not risen.",
  "Paul says preaching would remain useful if Christ had not risen.", "The resurrection happened on a Thursday morning."),
 ("2 Corinthians 12:9", "God says his grace is sufficient for Paul.",
  "God says his grace is insufficient for Paul.", "Every chronic disease disappears immediately after prayer."),
 ("Ephesians 2:8", "Salvation is described as God's gift through grace and faith.",
  "Salvation is described as being entirely earned without grace.", "Every academic scholarship should be rejected."),
 ("Philippians 2:3", "People are told to avoid acting out of vanity or selfish strife.",
  "People are told to make vanity the motive of every action.", "The passage forbids all competitive sports."),
 ("1 Peter 5:7", "People are told to cast their cares on God because he cares for them.",
  "People are told God has no care for them.", "All mental health conditions are caused by personal sin."),
 ("Revelation 22:12", "The speaker says he is coming quickly with his reward.",
  "The speaker says he will never come again.", "The second coming is dated to the year 2027."),
 ("Mark 4:39", "Jesus rebuked the wind, and the wind stopped.",
  "Jesus rebuked the wind, but the wind became stronger.", "Weather forecasts are always useless."),
 ("Job 42:2", "Job says that God can do everything.",
  "Job says that God cannot do anything.", "Humans can accurately predict every future event."),
 ("Daniel 6:16", "The king ordered Daniel to be put into the den of lions.",
  "The king ordered Daniel never to be put into the den of lions.", "The lions were all vegetarian."),
 ("Mark 4:39–40", "Jesus calmed the wind and questioned the disciples' fear.",
  "Jesus failed to calm the storm and encouraged the disciples to panic.", "The boat contained a GPS satellite receiver."),
 ("Isaiah 55:8–9", "God's thoughts and ways are portrayed as higher than human thoughts and ways.",
  "God's thoughts and ways are described as lower than human thoughts and ways.", "This passage supplies the altitude of the atmosphere."),
 ("Romans 6:22–23", "The passage contrasts eternal life as God's gift with death as sin's wages.",
  "The passage says sin pays eternal life as its wages.", "This passage gives the exact date of a future election."),
 ("Acts 20:34–35", "The passage encourages supporting the weak through work and giving.",
  "The passage condemns helping the weak and forbids giving.", "The passage specifies the modern minimum hourly wage."),
]


def build(corpus: Path, train: Path, seed: Path, out: Path):
    passages = {p.reference:p for p in load_passages(corpus)}
    trained = [passages[x["reference"]] for x in
               (json.loads(row) for row in train.read_text().splitlines())]
    seeded = [passages[x["reference"]] for x in
              (json.loads(row) for row in seed.read_text().splitlines())]
    rows=[]
    for ref, support, contradict, neutral in PROBES:
        passage=passages[ref]
        if any(overlapping_refs(passage, x) for x in trained + seeded):
            raise ValueError(f"Holdout leaks near another sample: {ref}")
        for claim,label in ((support,0),(neutral,1),(contradict,2)):
            rows.append({"premise":passage.text,"hypothesis":claim,
                         "label":label,"reference":ref})
    if len(rows) < 90:
        raise ValueError("Expected at least 90 held-out examples")
    out.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    print("Wrote",len(rows),"holdout cases from",len(PROBES),"passages")


if __name__=="__main__":
    build(Path("data/en_kjv.json"),Path("training/domain_pairs.jsonl"),
          Path("evaluation/seed.jsonl"),Path("evaluation/fresh_holdout.jsonl"))
