using Printf

const OUTPUT_DIR = "decision_tree_output"
input_tex = joinpath(OUTPUT_DIR, "decision_tree.tex")          # Your original tex file
output_tex = joinpath(OUTPUT_DIR, "decision_tree_styled.tex")  # New styled output

# Read original .tex content
tex_content = read(input_tex, String)

# Your custom TikZ styles (from your final code, formatted as raw string)
tikz_node_style = raw"""
\tikzset{
  % Decision node style
  decision/.style={
    rectangle,
    draw=darkbluegray,
    fill=lightbluegray,
    minimum height=12mm,
    minimum width=30mm,
    inner sep=5pt,
    font=\\footnotesize\\sffamily,
    align=center,
    rounded corners=3pt,
    line width=0.8pt,
    drop shadow={
      shadow xshift=1pt,
      shadow yshift=-1pt,
      opacity=0.15,
      fill=shadowcolor
    }
  },
  % Terminal node style (for routes)
  terminal/.style={
    ellipse,
    draw=darkbluegray,
    fill=lightbluegray!70,
    minimum height=10mm,
    minimum width=28mm,
    inner sep=4pt,
    font=\\footnotesize\\sffamily\\bfseries,
    align=center,
    line width=0.8pt,
    drop shadow={
      shadow xshift=1pt,
      shadow yshift=-1pt,
      opacity=0.15,
      fill=shadowcolor
    }
  },
  % Edge style
  every edge/.style={
    draw=darkbluegray,
    line width=0.6pt,
    -stealth
  }
}
"""

# Split the tex file by lines
lines = split(tex_content, '\n')

# Find the last \usepackage or \usetikzlibrary line for insertion point
insert_index = findlast(x -> occursin("usetikzlibrary", x) || occursin("usepackage", x), lines)

# If not found, default to top of file
if insert_index === nothing
    insert_index = 1
end

# Insert your tikz_node_style after that line
new_lines = vcat(
    lines[1:insert_index],
    [tikz_node_style],
    lines[insert_index+1:end]
)

# Write the modified content to new output file
open(output_tex, "w") do f
    for line in new_lines
        println(f, line)
    end
end

println("✓ Created styled .tex file at: $output_tex")
println("You can now compile this with xelatex or lualatex to see your styled decision tree.")