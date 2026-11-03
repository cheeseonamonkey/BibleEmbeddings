"""Curated Bible-anchored evidence pairs; authored propositions, programmatically expanded.

Each line is [verse range, supported claim, explicitly opposed claim, nearby
but unproven claim]. Labels characterize the TEXT, not historical truth.
No examples are taken from evaluation/seed.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from bible_embeddings.corpus import load_passages

# Verified against the cited KJV verses. Three contrasting claims per anchor.
# Deliberately includes awkward modern claims and close topical non sequiturs.
CASES = [
 ("Genesis 1:1", "God created the heaven and the earth.", "The Bible says God did not create the heaven and earth.", "The universe is exactly 13.8 billion years old."),
 ("Genesis 2:7", "God formed man from the dust of the ground.", "This passage says man was not formed from dust.", "The first man had blue eyes."),
 ("Genesis 2:18", "God said it was not good for the man to be alone.", "God said it was good for the man to be alone.", "Every person must marry before turning thirty."),
 ("Genesis 3:19", "The man is told he will return to dust.", "The man is told he will never return to dust.", "The age of the earth is specified in years."),
 ("Genesis 6:14", "Noah is told to make an ark from gopher wood.", "Noah is told not to build an ark.", "A modern cargo ship must be made of wood."),
 ("Genesis 9:13", "God sets a rainbow in the cloud as a covenant sign.", "God says the rainbow cannot be a sign of his covenant.", "Rainbows only appear in tropical climates."),
 ("Genesis 11:6", "The people are described as sharing one language.", "The people are described as speaking many different languages.", "The tower was exactly 200 meters tall."),
 ("Genesis 12:2", "God promises to make Abram into a great nation.", "God promises never to make Abram into a nation.", "Abram's descendants will all be wealthy."),
 ("Genesis 37:3", "Israel loved Joseph more than his other children.", "Israel hated Joseph more than his other children.", "Favoritism is always permitted in every family."),
 ("Exodus 3:5", "Moses is ordered to remove his shoes on holy ground.", "Moses is instructed to keep his shoes on while standing on holy ground.", "Shoes are forbidden in all places of worship today."),
 ("Exodus 3:14", "God answers Moses with the words I AM THAT I AM.", "God denies saying I AM THAT I AM to Moses.", "Moses speaks fluent Latin."),
 ("Exodus 14:21", "The waters of the sea were divided.", "The sea remained completely undivided.", "The sea crossing happened on a Tuesday."),
 ("Exodus 16:4", "God promises bread from heaven for the people.", "God explicitly refuses to give the people bread from heaven.", "Heavenly bread contains precisely 200 calories."),
 ("Exodus 20:3", "The commandment forbids having other gods before the LORD.", "The commandment encourages having other gods before the LORD.", "All modern governments must prohibit non-Abrahamic religions."),
 ("Exodus 20:8", "The Sabbath day is to be remembered and kept holy.", "The Sabbath is commanded to be forgotten and treated as unholy.", "The workweek must legally be four days."),
 ("Leviticus 19:18", "The passage commands love of one's neighbor as oneself.", "The passage commands hatred of one's neighbor.", "Neighbors must always share a political party."),
 ("Leviticus 19:34", "Israelites are told to love the stranger living among them.", "Israelites are told to hate every stranger among them.", "Every nation must abolish all immigration procedures."),
 ("Leviticus 19:35", "The passage condemns dishonesty in measures and weights.", "The passage orders people to use dishonest weights.", "All retailers must use digital scales."),
 ("Numbers 6:24", "The blessing asks the LORD to keep and bless the hearer.", "The blessing asks the LORD to curse the hearer.", "The recipient is guaranteed a long vacation."),
 ("Numbers 22:28", "The LORD opens the donkey's mouth so it can speak.", "The donkey is explicitly described as unable to speak.", "All donkeys can speak human languages today."),
 ("Deuteronomy 6:5", "The people are told to love God with all their heart and soul.", "The people are told never to love God.", "Love of God can be measured with an electronic sensor."),
 ("Deuteronomy 8:3", "The passage teaches that people do not live by bread alone.", "The passage says people live by bread alone.", "Bread is a prohibited food for Christians."),
 ("Deuteronomy 15:11", "The poor will not cease from the land, according to the passage.", "The poor will completely disappear from the land.", "A specific modern tax rate is required."),
 ("Deuteronomy 24:17", "The text forbids perverting judgment against strangers and the fatherless.", "The text commands judges to treat fatherless people unfairly.", "Every court case must be heard by twelve jurors."),
 ("Joshua 1:9", "Joshua is instructed to be strong and courageous.", "Joshua is commanded to be afraid and discouraged.", "Joshua led a navy of 100 ships."),
 ("Ruth 1:16", "Ruth tells Naomi that she will go wherever Naomi goes.", "Ruth tells Naomi that she refuses to go with her.", "Ruth's journey required a train ticket."),
 ("1 Samuel 16:7", "The LORD says humans look at outward appearance but God looks at the heart.", "The LORD says God judges only outward appearance.", "Clothing choices are irrelevant in every human relationship."),
 ("Job 1:21", "Job says the LORD gave and has taken away.", "Job says the LORD neither gives nor takes away.", "Job owned a fleet of electric cars."),
 ("Job 38:4", "God asks Job where he was when the earth's foundations were laid.", "God says Job built the foundations of the earth.", "The precise date of earth's formation is provided."),
 ("Psalms 14:1", "The psalm calls one who says there is no God a fool.", "The psalm praises the one who says there is no God as wise.", "Every atheist must be sentenced by a civil court."),
 ("Psalms 23:1", "The psalmist calls the LORD his shepherd.", "The psalmist denies that the LORD is his shepherd.", "The psalmist owned exactly ten sheep."),
 ("Psalms 46:10", "The speaker says to be still and know that he is God.", "The speaker says nobody should ever be still.", "Meditation apps are divinely mandated."),
 ("Psalms 90:12", "The psalm asks to be taught to number our days.", "The psalm asks that people never consider the passing of days.", "The text predicts the year 2099."),
 ("Psalms 119:105", "God's word is compared to a lamp and a light.", "God's word is called darkness rather than light.", "This passage explains the operation of electric lamps."),
 ("Proverbs 3:5", "The proverb instructs people to trust in the LORD wholeheartedly.", "The proverb instructs people never to trust in the LORD.", "All scientific research should be legally prohibited."),
 ("Proverbs 6:6", "A lazy person is advised to observe the ant.", "A lazy person is forbidden to learn from ants.", "Ant colonies operate democratic elections."),
 ("Proverbs 10:12", "Hatred is described as stirring up strife.", "Hatred is described as preventing strife.", "Arguments are never useful in any situation."),
 ("Proverbs 12:22", "The LORD detests lying lips, according to the proverb.", "The LORD delights in lying lips.", "Every false statement is a punishable criminal offense."),
 ("Proverbs 15:1", "A gentle response is said to turn away wrath.", "A gentle response is said to provoke greater wrath.", "Soft-spoken people always win elections."),
 ("Proverbs 16:18", "Pride is said to precede destruction.", "Pride is said to prevent destruction.", "Every confident person will lose their job."),
 ("Proverbs 17:17", "The proverb says a friend loves at all times.", "The proverb says a friend never loves at any time.", "All friendships must be maintained through social media."),
 ("Proverbs 22:6", "The proverb advises training a child in the way he should go.", "The proverb advises never teaching children how to live.", "Children should all attend the same type of modern school."),
 ("Proverbs 25:21", "The reader is told to feed an enemy who is hungry.", "The reader is told to deny bread to a hungry enemy.", "All restaurants must give everyone a free meal."),
 ("Ecclesiastes 3:1", "The passage says there is a time for every purpose under heaven.", "The passage says there is never a time for any purpose.", "The universe is governed by a digital clock."),
 ("Ecclesiastes 3:2", "The text says there is a time to be born and a time to die.", "The text says nobody ever dies.", "The exact year of an individual's death can be calculated."),
 ("Isaiah 1:17", "The text commands seeking justice and helping the oppressed.", "The text commands people to oppress widows.", "A particular contemporary voting system is divinely required."),
 ("Isaiah 9:6", "The passage says a child is born and a son is given.", "The passage says no child or son will be given.", "The passage identifies an exact modern birth certificate number."),
 ("Isaiah 41:10", "God says not to fear, for he is with the hearer.", "God says the hearer should be afraid because God will abandon them.", "Everyone who prays will win the lottery."),
 ("Jeremiah 1:5", "God says he knew Jeremiah before Jeremiah was formed in the womb.", "God says he only came to know Jeremiah after his birth.", "Every human existed consciously before birth."),
 ("Jeremiah 17:9", "The heart is described as deceitful.", "The heart is described as wholly incapable of deceit.", "Doctors should stop studying the human heart."),
 ("Ezekiel 18:4", "The passage says the soul that sins shall die.", "The passage says the soul that sins will never die.", "An insurance premium should depend on the owner's religion."),
 ("Hosea 6:6", "The speaker prefers mercy and knowledge of God to sacrifices.", "The speaker says he prefers sacrifices to mercy.", "A modern church must close its doors to visitors."),
 ("Micah 6:8", "The LORD is said to require justice, mercy, and humility.", "The LORD is said to require cruelty and injustice.", "The passage mandates a particular modern welfare program."),
 ("Jonah 1:17", "Jonah spends three days and nights inside a great fish.", "Jonah is said not to have entered a great fish.", "The great fish was definitely a blue whale."),
 ("Nahum 1:7", "The LORD is described as good and as a refuge in trouble.", "The LORD is described as evil and powerless during trouble.", "Every believer is immune from natural disasters."),
 ("Habakkuk 2:4", "The just are said to live by faith.", "The just are said never to live by faith.", "The passage endorses a particular economic policy."),
 ("Matthew 5:9", "Jesus calls peacemakers blessed.", "Jesus calls peacemakers cursed.", "All peace treaties are guaranteed to succeed."),
 ("Matthew 5:16", "The hearers are told to let their good works be seen.", "The hearers are instructed never to do good works.", "Good works must always be broadcast on television."),
 ("Matthew 5:28", "Jesus connects lustful looking with adultery in the heart.", "Jesus says lustful looking has no connection to adultery.", "Every private thought should be monitored by government."),
 ("Matthew 6:19", "Jesus warns against storing up treasures on earth.", "Jesus orders everyone to store up treasure on earth.", "Using a savings account is always sinful."),
 ("Matthew 7:7", "Jesus instructs the audience to ask, seek, and knock.", "Jesus forbids the audience from asking or seeking.", "Every request made to another person is always fulfilled."),
 ("Matthew 7:12", "The hearer is told to treat others as they want to be treated.", "The hearer is told to treat others in ways they would hate.", "All social interactions must be recorded by law."),
 ("Matthew 7:15", "Jesus warns against false prophets.", "Jesus commands his followers to trust every false prophet.", "The false prophets all have the same clothing brand."),
 ("Matthew 22:21", "Jesus says to give Caesar what belongs to Caesar.", "Jesus says nothing whatsoever should be given to Caesar.", "All governments must charge an identical income tax."),
 ("Matthew 22:39", "The command is to love one's neighbor as oneself.", "The command is to despise one's neighbor.", "This text specifies the minimum wage."),
 ("Mark 2:27", "Jesus says the Sabbath was made for man.", "Jesus says man was made solely for the Sabbath.", "The Sabbath must begin at 6:00 PM in every country."),
 ("Mark 10:14", "Jesus instructs that children be allowed to come to him.", "Jesus commands that children be forbidden to approach him.", "Every school should be funded by the state."),
 ("Luke 18:25", "Jesus compares a rich person's entry into God's kingdom to a camel going through a needle.", "Jesus says wealth guarantees entry into God's kingdom.", "Everyone with more than a million dollars is necessarily wicked."),
 ("John 1:1", "The Word is described as being with God and being God.", "The Word is explicitly said to have nothing to do with God.", "The passage identifies the Word's language as French."),
 ("John 3:16", "The passage says God loved the world.", "The passage says God hated the world.", "All humans will live exactly 120 years."),
 ("John 8:32", "Jesus says knowing the truth will make his hearers free.", "Jesus says truth can never bring freedom.", "The internet makes every person free."),
 ("John 13:34", "Jesus commands his followers to love one another.", "Jesus commands his followers to hate one another.", "Every follower must belong to a particular political party."),
 ("John 14:6", "Jesus calls himself the way, the truth, and the life.", "Jesus explicitly denies being the way and the truth.", "Jesus says every road in Rome leads to heaven."),
 ("Acts 2:38", "Peter tells the audience to repent and be baptized.", "Peter commands the audience never to repent or be baptized.", "Baptism must be performed only in a swimming pool."),
 ("Romans 3:23", "The passage says all have sinned.", "The passage says nobody has sinned.", "Every sin is punishable by a civil fine."),
 ("Romans 5:8", "The passage says Christ died for people while they were sinners.", "The passage denies that Christ died for sinners.", "This passage tells the date on which Jesus died."),
 ("Romans 12:18", "The reader is advised to live peaceably with all people when possible.", "The reader is told to quarrel with all people whenever possible.", "The law must ban all disagreements."),
 ("Romans 12:21", "The reader is commanded to overcome evil with good.", "The reader is commanded to overcome good with evil.", "Every good deed will produce an immediate reward."),
 ("1 Corinthians 10:13", "The passage says God is faithful amid temptation.", "The passage says God is unfaithful amid temptation.", "The verse proves humans can never feel temptation."),
 ("2 Corinthians 5:7", "Paul says believers walk by faith rather than sight.", "Paul says believers walk by sight rather than faith.", "This verse forbids wearing glasses."),
 ("Galatians 5:22", "Love, joy, and peace are described as fruit of the Spirit.", "The fruit of the Spirit is described as hatred and chaos.", "Every believer must have a cheerful personality."),
 ("Ephesians 4:26", "The readers are told not to sin in their anger.", "The readers are commanded to sin when angry.", "All anger is illegal."),
 ("Ephesians 4:29", "The readers are told not to speak corrupt words.", "The readers are instructed to speak corrupt words.", "Every joke is sinful."),
 ("Philippians 4:6", "The hearers are encouraged to pray rather than be anxious.", "The hearers are forbidden to pray about their concerns.", "Prayer always eliminates medical anxiety disorders."),
 ("Colossians 3:23", "The passage advises wholehearted work.", "The passage advises doing every task unwillingly and halfheartedly.", "Working from home is prohibited."),
 ("1 Thessalonians 5:17", "The readers are instructed to pray without ceasing.", "The readers are ordered never to pray.", "The passage mandates a particular daily meditation app."),
 ("1 Timothy 4:12", "Timothy is told to set an example despite his youth.", "Timothy is instructed to be ashamed of being young.", "Everyone under thirty should become a minister."),
 ("2 Timothy 3:16", "The passage describes scripture as inspired by God.", "The passage describes scripture as completely uninspired by God.", "Every translation has exactly the same number of words."),
 ("Hebrews 11:1", "Faith is described as the substance of things hoped for.", "Faith is described as hope's complete absence.", "Faith can be measured in kilograms."),
 ("Hebrews 13:2", "The hearers are told not to neglect hospitality to strangers.", "The hearers are told to refuse all hospitality to strangers.", "Every stranger is secretly an angel."),
 ("James 1:5", "A person lacking wisdom is told to ask God.", "A person lacking wisdom is forbidden to ask God.", "Every question will receive an immediate audible answer."),
 ("James 1:19", "The passage urges people to listen quickly and be slow to anger.", "The passage urges people to be quick to anger and slow to listen.", "All people should remain silent at public meetings."),
 ("1 John 4:8", "The passage says God is love.", "The passage says God is hatred.", "Love is a chemical element."),
 ("1 John 4:18", "Perfect love is said to cast out fear.", "Perfect love is said to produce fear.", "Anyone who loves another person can never feel fear."),
 ("Revelation 21:4", "The vision describes an end to death and pain.", "The vision predicts increasing death and pain forever.", "The end of pain will happen in the year 2050."),
 # Multi-verse premises intentionally encode an idea across adjacent verses.
 ("Matthew 6:19–20", "The audience is told to store treasures in heaven rather than on earth.", "Jesus commands his followers to store all treasures only on earth.", "Using a bank account is forbidden."),
 ("Matthew 7:7–8", "The audience is invited to ask, seek, and knock.", "The audience is forbidden to ask, seek, or knock.", "Every worldly request receives exactly the requested result."),
 ("Proverbs 15:1–2", "A gentle reply is praised while harsh words are associated with anger.", "The text praises harsh speech as the sure way to end anger.", "All anger is always unjustified."),
 ("Proverbs 6:6–8", "A lazy person is encouraged to learn diligence from the ant.", "A lazy person is told to ignore ants and never work.", "Ants store food in electric refrigerators."),
 ("Ecclesiastes 3:1–2", "The text says there is a time for birth, death, planting, and uprooting.", "The text says there is no time for either birth or death.", "Life expectancy is exactly eighty years."),
 ("Genesis 1:1–3", "The account describes God creating the world and calling light into existence.", "The account says God neither created the world nor spoke of light.", "The text gives the exact speed of light in meters per second."),
 ("Exodus 20:3–4", "The commandments prohibit other gods and making graven images.", "The commandments endorse worshipping other gods.", "All photographs should be illegal."),
 ("Psalms 23:1–2", "The psalmist pictures the LORD as a shepherd who leads him.", "The psalmist says the LORD refuses to guide him.", "The shepherd owns a modern sheep ranch."),
 ("Mark 10:13–14", "Jesus tells the disciples not to keep children from him.", "Jesus orders children to be kept away from him.", "Every child is required to attend a Sunday school."),
 ("Romans 12:17–18", "The text favors peace rather than repaying evil with evil.", "The passage orders readers to repay every evil act with evil.", "All wars are explicitly declared illegal."),
 ("Philippians 4:6–7", "The passage connects prayer and thanksgiving with the peace of God.", "The passage tells people never to pray in anxious circumstances.", "Every medical condition is healed by prayer."),
 ("James 1:19–20", "The passage warns against human wrath and commends patient listening.", "The passage praises human wrath as God's righteousness.", "Anger is never useful under any circumstances."),
 ("Galatians 5:22–23", "Love, joy, peace, and self-control are named as fruit of the Spirit.", "Cruelty and hate are listed as the Spirit's main fruit.", "All cheerful people are Christians."),
 ("1 John 4:7–8", "The writer urges mutual love and says God is love.", "The writer commands hatred and denies God is love.", "Every affectionate relationship is a marriage."),
]

# Hard-neutral statements: explicit semantic resemblance must not become
# evidence. The references are different from the 30-case development seed.
NEUTRALS = [
 ("Genesis 1:1", "The Bible reveals the exact age of the universe in seconds."),
 ("Genesis 1:1", "Scientists will discover a sixth fundamental force next year."),
 ("Genesis 2:7", "This passage proves every human is exactly the same height."),
 ("Genesis 9:13", "The rainbow is exclusively a modern political symbol."),
 ("Exodus 20:3", "The law should criminalize all non-Christian beliefs."),
 ("Exodus 20:8", "The United States must change the weekend to three days."),
 ("Leviticus 19:34", "A particular modern country's immigration policy is legally invalid."),
 ("Numbers 22:28", "Every domesticated animal can communicate in English."),
 ("Deuteronomy 15:11", "The maximum permissible income tax rate is 18%."),
 ("Deuteronomy 24:17", "A modern judge has committed bribery."),
 ("Psalms 14:1", "All people without religious beliefs are criminals."),
 ("Proverbs 6:6", "Ant colonies have advanced artificial intelligence."),
 ("Proverbs 12:22", "The politician in the last election told a lie."),
 ("Proverbs 15:1", "A soft voice is always evidence that its speaker tells the truth."),
 ("Proverbs 16:18", "Confidence and pride are interchangeable in every language."),
 ("Isaiah 1:17", "A particular candidate was elected by fraud."),
 ("Isaiah 41:10", "A person's exact lifespan can be predicted from their prayers."),
 ("Jeremiah 17:9", "Heart surgery is inherently wicked."),
 ("Matthew 5:9", "A peace treaty was signed yesterday."),
 ("Matthew 5:28", "The passage gives a universal criminal sentence for lust."),
 ("Matthew 6:19–20", "All bank accounts are forbidden by divine command."),
 ("Matthew 7:7–8", "If a person asks for a sports car they will receive one."),
 ("Matthew 7:12", "This verse proves a particular constitution should be rewritten."),
 ("Matthew 22:21", "The government may levy exactly five percent sales tax."),
 ("Luke 18:25", "Every wealthy individual is an evil person."),
 ("John 8:32", "A computer program can detect the truth of any political claim."),
 ("John 13:34", "One specific political leader secretly planned a disaster."),
 ("Romans 3:23", "Every modern criminal conviction is justified."),
 ("Romans 12:18", "The government must pass a law prohibiting all arguments."),
 ("1 Corinthians 10:13", "Nobody will ever suffer psychological distress."),
 ("2 Corinthians 5:7", "Eyeglasses are contrary to faith."),
 ("Galatians 5:22–23", "Every extroverted person has the Holy Spirit."),
 ("Philippians 4:6–7", "Taking medicine for anxiety is wrong."),
 ("Hebrews 11:1", "Faith is a measurable field with units of joules."),
 ("1 John 4:8", "Love has exactly twelve chemical components."),
 ("Revelation 21:4", "The end of suffering will occur on December 12, 2100."),
]


def overlapping_refs(a, b) -> bool:
    return a.book == b.book and a.chapter == b.chapter and a.start <= b.end + 2 and b.start <= a.end + 2


def make(corpus: Path, evaluation: Path, output: Path) -> dict:
    passages = {p.reference: p for p in load_passages(corpus)}
    holdout = {json.loads(s)["reference"] for s in evaluation.read_text().splitlines() if s.strip()}
    held = [passages[x] for x in holdout]
    examples, seen = [], set()

    def append(ref, claim, label, kind):
        passage = passages[ref]
        if any(overlapping_refs(passage, x) for x in held):
            raise ValueError(f"Training passage near existing evaluation: {ref}")
        key = (passage.text.casefold(), claim.casefold())
        if key in seen:
            return
        seen.add(key)
        examples.append({
            "premise": passage.text, "hypothesis": claim,
            "label": ("entailment", "neutral", "contradiction").index(label),
            "reference": ref, "kind": kind
        })

    for ref, supported, contradicted, neutral in CASES:
        for claim, label, kind in (
            (supported, "entailment", "literal_or_paraphrase"),
            (contradicted, "contradiction", "explicit_opposition"),
            (neutral, "neutral", "topical_non_sequitur"),
        ):
            append(ref, claim, label, kind)
    for ref, claim in NEUTRALS:
        append(ref, claim, "neutral", "hard_neutral")

    if len(examples) < 280:
        raise ValueError("Insufficient curated training examples")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in examples))
    result = {
        "count": len(examples),
        "labels": dict(Counter(("entailment", "neutral", "contradiction")[x["label"]] for x in examples)),
        "kinds": dict(Counter(x["kind"] for x in examples)),
        "references": len({x["reference"] for x in examples}),
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "note": "Author-crafted claims + deterministic labels, not independently annotated or validated.",
    }
    (output.parent / "domain_manifest.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=Path("data/en_kjv.json"))
    p.add_argument("--evaluation", type=Path, default=Path("evaluation/seed.jsonl"))
    p.add_argument("--out", type=Path, default=Path("training/domain_pairs.jsonl"))
    args = p.parse_args()
    print(json.dumps(make(args.corpus, args.evaluation, args.out), indent=2))
