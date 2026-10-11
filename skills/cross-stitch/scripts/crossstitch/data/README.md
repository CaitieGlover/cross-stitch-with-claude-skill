# DMC thread table

`dmc.csv` lists 454 DMC six-strand cotton colours with an RGB approximation
for each (`number,name,hex`).

Source: the widely shared "DMC Cotton Floss converted to RGB Values" table,
taken from the copy in
[adrianj/CrossStitchCreator](https://github.com/adrianj/CrossStitchCreator/blob/master/CrossStitchCreator/Resources/DMC%20Cotton%20Floss%20converted%20to%20RGB%20Values.csv).
Changes made here: columns reduced to number, name and hex; hex rebuilt from
the R, G, B columns (the source's hex for 310 was `0`); three names with a
lost dash character restored (606 Orange-Red Bright, 919 Red-Copper,
918 Red-Copper Dark).

RGB values for thread are approximations. Different published tables
disagree by a few units, and real floss looks different under different
light, so treat matches as a starting point. The table predates DMC's 2017
colours numbered 01-35, which are not included.
