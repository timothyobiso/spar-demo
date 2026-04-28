"""Train CAA-style period-specific steering vectors.

For each named period (tonight, tomorrow, one_week, one_month, one_year,
a_decade), compute mean-pooled residual-stream activation across a corpus of
sentences anchored in that period, and contrast against a pooled mean of all
other-period + neutral sentences:

    direction(period, L) = μ(period @ L) − μ(other + neutral @ L)

Saving each as a separate (setname, target) entry so the UI shows one slider
per period. The legacy probes (log_time_horizon, planning_depth) are kept by
merging with an existing probes.pkl.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PERIODS: dict[str, list[str]] = {
    "tonight": [
        "I'll see you at the show tonight.",
        "We're having pasta tonight.",
        "The fireworks start at nine tonight.",
        "She's reading bedtime stories tonight.",
        "He plans to call his mother tonight.",
        "Tonight the moon will be full.",
        "The diner closes at eleven tonight.",
        "I'm staying in tonight.",
        "There's a thunderstorm forecast for tonight.",
        "The team is playing at home tonight.",
        "We're catching the late movie tonight.",
        "Tonight feels quiet on the street.",
        "She finishes the report tonight.",
        "The concert is tonight at the park.",
        "He'll lock the gate before bed tonight.",
        "Tonight's special is grilled salmon.",
        "I want to be asleep by midnight tonight.",
        "Tonight we light the candles for dinner.",
        "The kids are at a sleepover tonight.",
        "I left the porch light on tonight.",
        "Tonight the wind has finally died down.",
        "We're meeting for drinks tonight.",
        "Don't forget to take out the trash tonight.",
        "Tonight is the launch party.",
        "He's working the late shift tonight.",
    ],
    "tomorrow": [
        "I'll send the draft tomorrow morning.",
        "We're flying to Lisbon tomorrow.",
        "The exam is at nine tomorrow.",
        "Tomorrow the painters arrive.",
        "She has a dentist appointment tomorrow.",
        "By tomorrow evening the package should arrive.",
        "Tomorrow we begin the renovation.",
        "He plans to apologize tomorrow.",
        "Rain is forecast for tomorrow afternoon.",
        "I'll see you at the office tomorrow.",
        "Tomorrow is her birthday.",
        "We have a lunch meeting tomorrow.",
        "The bakery opens at six tomorrow.",
        "Tomorrow he turns thirty.",
        "I'll finish the chapter tomorrow.",
        "Tomorrow's lecture is on plate tectonics.",
        "She's running a marathon tomorrow.",
        "Tomorrow the grocery store delivers.",
        "We're all going hiking tomorrow.",
        "Tomorrow morning, please water the plants.",
        "He's leaving for college tomorrow.",
        "Tomorrow I have to renew my license.",
        "We're testing the prototype tomorrow.",
        "The bus schedule changes tomorrow.",
        "Tomorrow brings the first frost of autumn.",
    ],
    "one_week": [
        "She's flying out next week for the conference.",
        "The package will arrive within a week.",
        "We have a deadline in seven days.",
        "Next week I start the new job.",
        "The festival lasts an entire week.",
        "He'll be back from his trip in a week.",
        "Within a week the project should be done.",
        "We're hosting the team dinner next week.",
        "The dentist scheduled the cleaning for next week.",
        "I've blocked off all of next week for vacation.",
        "Within a week she had finished the manuscript.",
        "They give you a week to respond to the letter.",
        "Our weekly meeting runs every Friday.",
        "The semester ends in a week.",
        "Next week is performance review.",
        "He goes camping every weekend.",
        "By next week we'll know the results.",
        "I have one week of paid leave.",
        "Within seven days the bruise had faded.",
        "Next week the plumbers come.",
        "We have a week before the trip.",
        "She gave the manager one week's notice.",
        "Within a week the rumor had spread.",
        "Next week we're moving offices.",
        "The lease has one more week to run.",
    ],
    "one_month": [
        "We're moving in a month.",
        "The trial lasts a month.",
        "Within a month she had learned the basics.",
        "Next month is my parents' anniversary.",
        "The promotion starts in a month.",
        "By the end of next month the building should be done.",
        "He's gone for the entire month of August.",
        "Within thirty days the rebate arrives.",
        "Next month rent goes up.",
        "The book club meets the first Tuesday of every month.",
        "I have a doctor's appointment in a month.",
        "Within a month the tomatoes will ripen.",
        "Next month the new policy takes effect.",
        "The course is one month long.",
        "I've saved a month's expenses already.",
        "By next month we'll have a new logo.",
        "The internship is just over a month.",
        "Within four weeks she had the answer.",
        "Next month his grandfather visits.",
        "The construction wraps up in a month.",
        "I'll get a haircut next month.",
        "The seasonal staff is hired by the month.",
        "Within a month the seedlings reached six inches.",
        "Next month is the Lunar New Year.",
        "The contract renews every month.",
    ],
    "one_year": [
        "Next year I'll finally graduate.",
        "We're planning to travel to Japan next year.",
        "The lease is for one year.",
        "Within a year the company doubled its staff.",
        "Next year is an election year.",
        "By next year the facility will be complete.",
        "It's been almost a year since the move.",
        "I'll re-evaluate everything in a year.",
        "Within twelve months the seedling became a tree.",
        "Next year the championships are in Berlin.",
        "The fellowship runs for one year.",
        "By next April the bridge will be open.",
        "Within a year of buying it, the laptop broke.",
        "Next year my brother gets married.",
        "The data covers a single year.",
        "She gave herself a year to write the book.",
        "Within twelve months the loan is repaid.",
        "Next year the school adds a music wing.",
        "The forecast is for one full year.",
        "By next summer they will have moved.",
        "She runs one race a year.",
        "Within a year, the city had recovered.",
        "Next year my passport expires.",
        "The warranty covers the first year.",
        "I'll finish my thesis within a year.",
    ],
    "a_decade": [
        "Within a decade, the neighborhood had transformed.",
        "He spent ten years studying jazz piano.",
        "The treaty was signed a decade ago.",
        "By the late 2030s, electric vehicles dominated.",
        "The technology will mature over the next decade.",
        "Their friendship spanned a decade.",
        "Ten years of work went into the bridge.",
        "Within a decade, the species had returned.",
        "The decade after the war was unstable.",
        "He built the company over ten years.",
        "By 2040, climate adaptation will be widespread.",
        "She lived in the apartment for a decade.",
        "Within ten years, the policy had reshaped housing.",
        "The decade-long renovation finished last spring.",
        "Ten years from now, this skyline will be unrecognizable.",
        "By the end of the decade, the sport had grown global.",
        "He recorded ten albums over the decade.",
        "The decade of the 1990s brought rapid change.",
        "Within ten years she became chief surgeon.",
        "By the next decade, AI tools will be ubiquitous.",
        "The decade closed with an economic boom.",
        "He served on the board for a decade.",
        "Within ten years the river ecosystem recovered.",
        "Her decade in academia shaped the field.",
        "Ten years was enough to forget the language.",
    ],
}

SHORT_HORIZON = [
    "She finished her coffee in three minutes.",
    "The phone call lasted only a few seconds.",
    "Within an hour the snow had melted.",
    "He glanced at the menu for a brief moment.",
    "The whole conversation took five minutes.",
    "By noon the meeting was already over.",
    "She solved the puzzle in seconds.",
    "The rain stopped after a quarter hour.",
    "He blinked, and the bird was gone.",
    "Tea steeps for about three minutes.",
    "The egg boils in ten minutes.",
    "Within seconds the alarm fell silent.",
    "The lecture lasted only a half hour.",
    "He waited a few moments and tried again.",
    "Their handshake lasted just a second.",
    "Within minutes the reply came back.",
    "It's a three-minute walk to the station.",
    "He read the email in under a minute.",
    "The crowd dispersed in a matter of minutes.",
    "She made the decision in an instant.",
    "Lunch rushed by in twenty minutes.",
    "He shut the laptop after half an hour.",
    "The pop song lasts three minutes.",
    "Within seconds the message had spread through the room.",
    "It's a quick errand — twenty minutes there and back.",
    "By the time she looked up, only ten minutes had passed.",
    "He gave a five-minute toast.",
    "The pasta cooks in under nine minutes.",
    "She spent only a moment reviewing the document.",
    "An ambulance arrived within four minutes.",
    "The stoplight changes every thirty seconds.",
    "He read the headline in a glance.",
    "We had ten minutes before the train left.",
    "It only took her a minute to pack.",
    "The coffee was cold by the time he returned.",
    "It's a ten-minute meeting, no more.",
    "He hung up after a brief exchange.",
    "Within an hour, the news was everywhere.",
    "She closed her eyes for just a second.",
    "The kettle boils in three minutes.",
]

LONG_HORIZON = [
    "Her career spanned more than four decades.",
    "The project took seven years to complete.",
    "Over generations the family preserved the recipe.",
    "It took him most of a decade to write the book.",
    "The redwood has stood for nearly a thousand years.",
    "Their friendship lasted a lifetime.",
    "The cathedral was built over two centuries.",
    "She studied the language for fifteen years.",
    "The migration patterns developed over millennia.",
    "He ran the company for forty-five years.",
    "It took a generation for the law to change.",
    "The empire endured for three hundred years.",
    "Her research program ran for twenty years.",
    "Over the course of decades, the city transformed.",
    "He built the violin slowly over four years.",
    "The tradition has been kept for many centuries.",
    "Their marriage lasted fifty-eight years.",
    "The glacier had been receding for centuries.",
    "He spent thirty years perfecting the craft.",
    "Her doctorate took eight years of research.",
    "The novel was written over a decade.",
    "It took millions of years for the canyon to form.",
    "The university has trained scholars for six hundred years.",
    "She raised her children over twenty patient years.",
    "Across many generations, the language drifted.",
    "The dynasty ruled for four centuries.",
    "He served on the council for thirty-two years.",
    "Their correspondence lasted nearly four decades.",
    "The archive holds records spanning seven hundred years.",
    "Her scholarship matured over a long career.",
    "It took ten years for the seedling to bear fruit.",
    "The migration unfolded over thousands of years.",
    "She maintained the garden for fifty seasons.",
    "He apprenticed for seven years before mastery.",
    "The cathedral plans took two generations to realize.",
    "Their legal battle stretched over twelve years.",
    "The novel chronicles three generations.",
    "She held the post for twenty-eight years.",
    "He trained for the Olympics over a decade.",
    "The forest regenerated over the course of a century.",
]

NEUTRAL = [
    "The kitchen smelled of bread.",
    "He kept the book on the top shelf.",
    "Trees lined the road on both sides.",
    "Music drifted from the open window.",
    "The chair by the fireplace was hers.",
    "Her hand rested on the railing.",
    "Coffee stains marked the tablecloth.",
    "The hallway carpet was deep red.",
    "He preferred the quieter coffee shop.",
    "Books were stacked along the wall.",
    "The umbrella was leaning by the door.",
    "She laughed at the joke.",
    "Steam rose from the kettle.",
    "Crows called from the rooftop.",
    "The path turned sharply uphill.",
    "He tied his shoes carefully.",
    "Cinnamon and clove drifted from the kitchen.",
    "The blanket was thick and soft.",
    "Lanterns glowed along the porch.",
    "She liked the smell of new books.",
    "The bicycle leaned against the fence.",
    "He spoke softly when he was tired.",
    "The river curved past the village.",
    "A cat watched from the windowsill.",
    "Sunlight crossed the wooden floor.",
    "Pencils were scattered across the desk.",
    "The garden gate squeaked when opened.",
    "She drank her tea without sugar.",
    "He preferred wool sweaters in the cold.",
    "Footsteps echoed in the empty hall.",
]

DISPLAY_NAMES = {
    "tonight": "Tonight",
    "tomorrow": "Tomorrow",
    "one_week": "One Week",
    "one_month": "One Month",
    "one_year": "One Year",
    "a_decade": "A Decade",
}


def collect_pooled_activations(
    model, tokenizer, texts: list[str], device: str, layers: list[int]
) -> dict[int, np.ndarray]:
    blocks = model.model.layers
    captures: dict[int, list[np.ndarray]] = {l: [] for l in layers}

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, outputs):
            h = outputs[0] if isinstance(outputs, tuple) else outputs
            pooled = h[:, -1, :].squeeze(0).detach().to(torch.float32).cpu().numpy()
            captures[layer_idx].append(pooled)
        return hook

    handles = [blocks[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        for i, text in enumerate(texts):
            ids = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**ids)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(texts)}")
    finally:
        for h in handles:
            h.remove()

    return {l: np.stack(captures[l], axis=0) for l in layers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--out", default="results/qwen3.5-4b/probes_with_periods.pkl")
    parser.add_argument("--merge-with", default="results/qwen3.5-4b/probes.pkl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[probe] loading {args.model} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    )
    if device == "cuda":
        model = model.to("cuda")
    model.eval()

    n_layers = model.config.num_hidden_layers
    layers = list(range(0, n_layers, 2))
    if (n_layers - 1) not in layers:
        layers.append(n_layers - 1)
    print(f"[probe] hooking layers: {layers}")

    period_acts: dict[str, dict[int, np.ndarray]] = {}
    for name, sents in PERIODS.items():
        print(f"[probe] {name} (n={len(sents)})")
        period_acts[name] = collect_pooled_activations(model, tokenizer, sents, device, layers)

    print(f"[probe] neutral (n={len(NEUTRAL)})")
    neutral_acts = collect_pooled_activations(model, tokenizer, NEUTRAL, device, layers)

    print(f"[probe] short-horizon (n={len(SHORT_HORIZON)})")
    short_acts = collect_pooled_activations(model, tokenizer, SHORT_HORIZON, device, layers)
    print(f"[probe] long-horizon (n={len(LONG_HORIZON)})")
    long_acts = collect_pooled_activations(model, tokenizer, LONG_HORIZON, device, layers)

    out_probes: dict = {}
    if args.merge_with:
        merge_path = Path(args.merge_with)
        if merge_path.exists():
            with open(merge_path, "rb") as f:
                out_probes = pickle.load(f)
            print(f"[probe] merged with {merge_path}: {list(out_probes.keys())}")

    out_probes.setdefault("period_caa", {})
    for name in PERIODS:
        out_probes["period_caa"][name] = {}
        for l in layers:
            # contrast = mean of (all OTHER period sentences + neutral)
            other = [period_acts[n][l] for n in PERIODS if n != name]
            other.append(neutral_acts[l])
            contrast_mean = np.concatenate(other, axis=0).mean(axis=0)
            period_mean = period_acts[name][l].mean(axis=0)
            diff = period_mean - contrast_mean
            out_probes["period_caa"][name][l] = {
                "direction": diff.astype(np.float32),
            }
        norms = [
            float(np.linalg.norm(out_probes["period_caa"][name][l]["direction"]))
            for l in layers
        ]
        print(f"  {name}: layer norms min={min(norms):.2f} max={max(norms):.2f}")

    # Continuous long-vs-short horizon axis (CAA: μ_long − μ_short)
    out_probes.setdefault("horizon_caa", {})
    out_probes["horizon_caa"]["time_horizon"] = {}
    for l in layers:
        diff = long_acts[l].mean(axis=0) - short_acts[l].mean(axis=0)
        out_probes["horizon_caa"]["time_horizon"][l] = {
            "direction": diff.astype(np.float32),
        }
    norms = [
        float(np.linalg.norm(out_probes["horizon_caa"]["time_horizon"][l]["direction"]))
        for l in layers
    ]
    print(f"  long_vs_short: layer norms min={min(norms):.2f} max={max(norms):.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out_probes, f)
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
