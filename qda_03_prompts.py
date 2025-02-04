versioned_prompts: dict[int, dict[str, str]] = {
    1: {
        "phase1": """Your job is to analye a series of Reddit Comments are drawn from submissions related to the 2024 Queensland State Election. The comments are drawn from a collection of comments before, during and after the election itself on the 26th of October. The parties that sought election were the Australian Labor Party (ALP), the Liberal National Party (LNP), the Queensland Greens (Greens), One Nation and Katter's Australian Party. The LNP prevailed in the election, with David Crisafulli becoming the new Premier of Queensland. 
We have identified several key issues that the major parties campaigned on during the 2024 Queensland State Election: Youth Crime (YC), Cost of Living (COL), Health (H), Energy and Infrastructure (EI), and Abortion Laws (AL). Below is a summary of each party's position on these issues:
Youth Crime (YC):
- LNP: Advocated for stricter penalties for young offenders, including the 'Adult Crime, Adult Time' policy, proposing that serious offenses committed by youths be met with adult sentencing. 
- Labor: Focused on rehabilitation and prevention programs, aiming to address the root causes of youth crime through community engagement and support services.

Cost of Living (COL):
- LNP: Proposed measures such as abolishing stamp duty on new builds for first home buyers and re-establishing the productivity commission to review building industry regulations, aiming to alleviate housing costs. 
- Labor: Introduced initiatives like 50-cent public transport fares, a $1,000 energy rebate, and free lunches for state primary school students to ease cost-of-living pressures.

Health (H):
- LNP: Committed $590 million to an 'Easier Access to Health Services Plan,' aiming to reduce ambulance ramping and hire additional healthcare workers. 
- Labor: Focused on maintaining and expanding healthcare services, addressing hospital capacity issues, and investing in new health infrastructure.

Energy and Infrastructure (EI):
- LNP: Criticized Labor's Pioneer-Burdekin Pumped Hydro Project as financially unviable and canceled it upon taking office, opting to explore smaller-scale hydro projects instead. 
- Labor: Proposed large-scale renewable energy projects, including the Pioneer-Burdekin Pumped Hydro Project, to transition towards 80% renewable energy by 2035.

Abortion Laws (AL):
- LNP: Stated there would be 'no changes' to abortion laws under their government, maintaining the status quo established in 2018. 
- Labor Party: Supported the existing abortion laws and opposed any attempts to repeal or amend them. Asserts that LNP intend to restrict access to abortion services based on the views of some of their members.

Analyze Reddit comments and code them based on the positions they support or oppose concerning these issues. You will output JSON, so valid values are strings or null. Use the following codes: YC, COL, H, EI, AL. If a comment's position is unclear or does not pertain to these issues, output null.

Secondly identify which party the comment appears to support. Use the party code ALP, LNP, Greens, ON, KAP if this can be determined, otherwise a null value if undetermined.

Output JSON as a list of objects where each object has an "id" (integer) property of the comment id, an "issue" property set to the issue code string (or null value), and a "party" property set to the party code string or null value. Null means JSON null, not a string. Values MUST be quoted, e.g. {"party": "LNP"} unless they are null, e.g. {"party": null}. Comments follow:
""",
        "notes": "Gemini requires strong direction to output JSON correctly if we are using unquoted null values. If we were using a code like 'unknown' then it requires no such direction. However null values are more token efficient.",
    },
    2: {
        "phase1": """Your job is to analyse a series of Reddit comments from discussions related to the 2024 Queensland State Election. The major parties that sought election were the incumbent Australian Labor Party (ALP) and the Liberal National Party (LNP) opposition. The Queensland Greens (Greens), One Nation and Katter's Australian Party were minor parties in the election. 

The key five issues that the major parties campaigned on during the 2024 Queensland State Election were Youth Crime (YC), Cost of Living Crisis (COL), Health (H), Energy and Infrastructure (EI), and Abortion Laws (AL). We include some context of the major parties positions on these issues. 
Youth Crime (YC):
- LNP: Advocated for stricter penalties for young offenders, including the 'Adult Crime, Adult Time' policy, proposing that serious offenses committed by youths be met with adult sentencing. 
- Labor: Downplayed claimed youth crimewave. Focused on rehabilitation and prevention programs, aiming to address the root causes of youth crime through community engagement and support services.
Cost of Living Crisis (COL):
- LNP: Proposed measures such as abolishing stamp duty on new builds for first home buyers and re-establishing the productivity commission to review building industry regulations, aiming to alleviate housing costs. 
- Labor: Introduced initiatives like 50-cent public transport fares, a $1,000 energy rebate, and free lunches for state primary school students to ease cost-of-living pressures.
Health (H):
- LNP: Committed $590 million to an 'Easier Access to Health Services Plan,' aiming to reduce ambulance ramping and hire additional healthcare workers. 
- Labor: Focused on maintaining and expanding healthcare services, addressing hospital capacity issues, and investing in new health infrastructure.
Energy and Infrastructure (EI):
- LNP: Criticized Labor's Pioneer-Burdekin Pumped Hydro Project as financially unviable and canceled it upon taking office, opting to explore smaller-scale hydro projects instead. 
- Labor: Proposed large-scale renewable energy projects, including the Pioneer-Burdekin Pumped Hydro Project, to transition towards 80% renewable energy by 2035.
Abortion Laws (AL):
- LNP: Stated there would be 'no changes' to abortion laws under their government, maintaining the status quo established in 2018. 
- Labor Party: Supported the existing abortion laws and opposed any attempts to repeal or amend them. Asserts that LNP intend to restrict access to abortion services based on the views of some of their members.

Analyze the comments and code them according to two dimensions:
1. The ISSUE the commenter is primarily discussing, if it is one of the five listed above. Use the codes YC, COL, H, EI, AL. If no issue applies, omit the code entirely.
2. The PARTY the commenter appears to support. Use the party codes ALP, LNP, Greens, ON, KAP. Infer party support where possible, e.g. a commenter fearing the outcome of an LNP victory likely supports ALP, and vice versa. Ensure there is either direct or clearly inferable support for a party before assigning a party code. Assign a PARTY code only if the comment expresses explicit support for, or opposition to, a specific party or its policies. This support or opposition must be clearly discernible; mere mention of a policy is insufficient. Look for indicators such as positive or negative sentiment towards a party's actions or policies or explicit endorsements or condemnations. Comments may express opposition through sarcasm or irony. In cases where party support is not clear, omit the party code entirely.

You will output JSON as a list of objects where each object has an "id" (integer) property of the comment id, an "issue" property set to the issue code string, and a "party" property set to the party code string. In cases where you cannot determine an ISSUE or a PARTY, do not set a null value, omit the appropriate JSON properties entirely,  e.g. If the first comment did not have a matching issue and you could not determine party support, then the object would simply be {id: 1}.

Comments follow:
""",
        "notes": "Typo fixes. Removed info about LNP victory, added incumbent/opposition info,  frame major parties, and minor parties. Condensed text. Improved clarity of coding instructions and considerably elaborated on party support coding. Shifted to omitting JSON properties where not determined. Increased promimence of omit strategy (mentioned with coding and json output).",
    }
}
