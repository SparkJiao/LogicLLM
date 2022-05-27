from dataclasses import dataclass


@dataclass
class LSATDeductionExamples:
    data = [
        {
            "passage": "This morning, a bakery makes exactly one delivery, consisting of exactly six loaves of bread. "
                       "Each of the loaves is exactly one of three kinds: oatmeal, rye, or wheat, and each is either sliced or unsliced. "
                       "The loaves that the bakery delivers this morning must be consistent with the following: "
                       "There are at least two kinds of loaves. There are no more than three rye loaves. There is no unsliced wheat loaf. "
                       "There is at least one unsliced oatmeal loaf. "
                       "If two or more of the loaves are unsliced, then at least one of the unsliced loaves is rye.",
            "questions": [
                {
                    "question": "If the bakery delivers exactly four wheat loaves, then the bakery could also deliver?",
                    "positive_deductions": [
                        "One loaf is rye.",  # At least one loaf is rye or oatmeal.
                        "One loaf is oatmeal.",
                        "A loaf is oatmeal and unsliced.",
                    ],
                    "negative_deductions": [
                        "There are two rye loaves.",
                        "There are two unsliced rye loaves.",
                        "One sliced rye loaf and one unsliced rye loaf.",
                    ],
                }
            ],
            "positive_deductions": [],
            "negative_deductions": [],
        },
        {
            "passage": "Flyhigh Airlines owns exactly two planes: P and Q. "
                       "Getaway Airlines owns exactly three planes: R, S, T. "
                       "On Sunday, each plane makes exactly one flight, according to the following conditions: "
                       "Only one plane departs at a time. Each plane makes either a domestic or an international flight, but not both. "
                       "Plane P makes an international flight. Planes Q and R make domestic flights. "
                       "All international flights depart before any domestic flight. "
                       "Any Getaway domestic flight departs before Flyhigh's domestic flight.",
            "questions": [
                {
                    "question": "Which one of the following could be the order, from first to last, in which the five planes depart?",
                    "positive_deductions": [
                        "S makes international flights.",
                        "S makes international flights, T makes international flights.",
                        "S makes international flights, T makes international flights, S and T depart before R. R departs before Q.",
                        "S makes international flights, T makes domestic flights.",
                        "S makes international flights, T makes domestic flights, P departs before R. P departs before T. "
                        "T departs before Q.",
                    ],
                    "negative_deductions": [
                        "S makes domestic flights. T makes international flights. R departs before T.",
                        "S makes domestic flights. T makes international flights. S departs before T.",
                        "S makes domestic flights. T makes domestic flights. Q departs before S.",
                        "S makes domestic flights. T makes domestic flights. Q departs before T.",
                    ]
                }
            ],
            "positive_deductions": [
                "Plane P departs before planes Q and R.",
                "Plane R departs before plane Q.",
            ],
            "negative_deductions": [
                "Plane Q departs before plane P.",
                "Plane Q departs before plane R.",
            ],
        },
        {
            "passage": "Each of nine students, Faith, Gregory, Harlan, Jennifer, Kenji, Lisa, Marcus, Nari, and Paul, will be assigned to exactly one of three panels: Oceans, Recycling, and Wetlands. Exactly three of the students will be assigned to each panel. The assignment of students to panels must meet the following conditions: Faith is assigned to the same panel as Gregory. Kenji is assigned to the same panel as Marcus. Faith is not assigned to the same panel as Paul. Gregory is not assigned to the same panel as Harlan. Jennifer is not assigned to the same panel as Kenji. Harlan is not assigned to the Oceans panel if Paul is not assigned to the Oceans panel.",
            "questions": [
                {
                    "question": "If Kenji and Paul are both assigned to the Recycling panel, which one of the following could be true?",
                    "positive_deductions": [
                        "Mary is assigned to the Recycling panel.",
                        "Harlan is assigned to the Wetlands panel.",
                        "Gregory is not assigned to the Wetlands panel.",
                        "Gregory and Faith are assigned to the Oceans panel.",
                    ],
                    "negative_deductions": [
                        "Harlan is assigned to the Recycling panel.",
                        "Gregory is assigned to the Wetlands panel.",
                        "Mary is assigned to the Wetlands panel.",
                        "Faith is assigned to the Wetlands panel.",
                    ],
                }
            ],
            "positive_deductions": [
                "Faith and Gregory are assigned to the same panel with Jennifer, Lisa and Nari.",
                "Kenji and Marcus are assigned to the same panel with Harlan, Lisa, Nari and Paul.",
                "Faith and Harlan are assigned to different panel.",
            ],
            "negative_deductions": [
                "Gregory and Paul are assigned to the same panel.",
                "Mary and Jennifer are assigned to the same panel.",
                "Faith and Gregory are assigned to the same panel.",
            ]
        }
    ]


@dataclass
class LSATDeductionExamplesV2:
    data = [
        {
            "passage": "This morning, a bakery makes exactly one delivery, consisting of exactly six loaves of bread. "
                       "Each of the loaves is exactly one of three kinds: oatmeal, rye, or wheat, and each is either sliced or unsliced. "
                       "The loaves that the bakery delivers this morning must be consistent with the following: "
                       "There are at least two kinds of loaves. There are no more than three rye loaves. There is no unsliced wheat loaf. "
                       "There is at least one unsliced oatmeal loaf. "
                       "If two or more of the loaves are unsliced, then at least one of the unsliced loaves is rye.",
            "questions": [
                {
                    "question": "If the bakery makes exactly one delivery, which of the following is true?",
                    "positive_deductions": [
                        "There is a sliced oatmeal loaf.",
                        "There is a sliced rye loaf.",
                        "The four wheat loaves are unsliced.",
                    ],
                    "negative_deductions": [
                        "There are two unsliced oatmeal loaves.",
                        "There are two sliced oatmeal loaves.",
                    ],
                }
            ],
            "positive_deductions": [],
            "negative_deductions": [],
        },
    ]
