# Cover letter — ACS Omega

Dear Editors,

Please consider the enclosed manuscript, **"Objective-Dependent Active Learning and
Calibrated Uncertainty for Sample-Efficient Discovery of Rare-Earth Extractants: A
Benchmark on 1,202 Experimental Distribution Coefficients with Prospective
Validation,"** for publication in *ACS Omega*. The manuscript was transferred from the
*Journal of Chemical Information and Modeling* at the editor's invitation.

Measuring distribution coefficients (logD) for rare-earth solvent extraction is slow
and costly, so machine learning and active learning (AL) are increasingly applied to
prioritize experiments. This manuscript provides what that literature has largely
lacked: a rigorous, prospectively validated benchmark answering two practitioner
questions, namely *which* AL acquisition strategy to use, and *whether* the attendant
uncertainties can be trusted, on real experimental data.

The main contributions are:

1. **A prospectively validated benchmark** on 1,202 experimental logD measurements,
   with external validation on four independently synthesized ligands (R2 = 0.69).
   Prospective tests are rare in this area.
2. **The optimal acquisition is objective-dependent.** Exploitation recovers all top
   extractants but yields a poor predictive model; exploration/random gives the best
   model but finds few top extractants; no strategy wins both. We show this on **two
   independent datasets** (f-element extraction and a 4,200-compound logD set),
   establishing it as general rather than dataset-specific.
3. **A cautionary result**: for prediction, AL provides no advantage over random
   selection.
4. **A calibration distinction** that is easy to conflate: conformal calibration is
   essential for *trustworthy* uncertainty (8x better coverage) but does **not**
   improve AL acquisition, which depends only on the uncertainty ranking.

The work suits ACS Omega: it is technically rigorous and fully reproducible, built on
real experimental data, with prospective validation and cross-dataset generality, and
it offers practitioners an evidence-based decision guide rather than a single narrow
result. We are candid that it is a careful benchmark and analysis, not a new method
claimed to dominate prior alternatives; its value is rigor, generality, prospective
validation, and several non-obvious, practically useful findings.

All code and data are released and archived (Zenodo concept DOI
10.5281/zenodo.20764583); every reported number is reproducible on a single laptop.
The manuscript is original, not under consideration elsewhere, and use of an AI tool
is disclosed within. The author has no competing interests to declare.

Thank you for your consideration.

Sincerely,
Krishna Harish
Lawrence E. Elkins High School, Missouri City, Texas, USA
krishnaharish2009@gmail.com
