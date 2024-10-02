import argparse
import random
import os
import re
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./")
    parser.add_argument("--target_dir", type=str, default="./", help="the directory to save the processed data")
    parser.add_argument("--save_file", type=str, default="selected_paper_ids.txt", help="the file to save the selected paper ids")
    parser.add_argument("--num_papers", type=int, default=1000, help="the number of papers to select")
    parser.add_argument("--seed", type=int, default=42)
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    
    # read the csv files, and get the "keywords,tldr,abstract,track,acceptance,scores" fields
    CSV_FILE = ["ICLR_2023_draft_comment_meta.csv"]  # "ICLR_2022_draft_comment_meta.csv",
    # combine two csv files
    final_df = pd.DataFrame()
    for file in CSV_FILE:
        df = pd.read_csv(os.path.join(args.root_dir, file))
        df = df[["ID","keywords", "tldr", "abstract", "track", "acceptance", "scores"]]
        final_df = pd.concat([final_df, df])
    print(f"==> total {len(final_df)} papers")
    
    # 1. count the number of track and their distribution
    track_distribution = final_df["track"].value_counts()
    print("==> track distribution:")
    for track, count in track_distribution.items():
        print(f"{track}: {count}")
    
    # 2. count the number of acceptance and their distribution
    acceptance_distribution = final_df["acceptance"].value_counts()
    print("==> acceptance distribution:")
    for acceptance, count in acceptance_distribution.items():
        print(f"{acceptance}: {count}")
        
    # for the "acceptance" group, gather all the ["Accept: poster", "Accept: notable-top-25%", "Accept: notable-top-5%"] together
    acceptance_group = final_df[final_df["acceptance"].str.contains("Accept")]
    rej_group = final_df[final_df["acceptance"].str.contains("Reject")]
    # sort the acceptance_group by track
    acceptance_group = acceptance_group.sort_values(by=["track"])
    # sort the rejection group by track
    rej_group = rej_group.sort_values(by=["track"])
    print("==> acceptance group:" , len(acceptance_group))
    print("==> rejection group:" , len(rej_group))
    
    # for each group, evenly sample papers from all the tracks, each group 500 papers
    all_tracks = final_df["track"].unique()
    selected_acc_paper_ids = []
    # for the acceptance group, for loop whole group, for each track, pick one paper that are not in the selected_acc_paper_ids, then go for next track
    # until the number of selected papers reach 500
    num_acc_paper = args.num_papers // 2
    falg = True
    while falg:
        for track in all_tracks:
            track_group = acceptance_group[acceptance_group["track"]==track]
            for idx, row in track_group.iterrows():
                if row["ID"] not in selected_acc_paper_ids:
                    selected_acc_paper_ids.append(row["ID"])
                    break
            if len(selected_acc_paper_ids) >= num_acc_paper:
                falg = False
                break
    print(f"==> selected {len(selected_acc_paper_ids)} acceptance papers")
    
    selected_rej_paper_ids = []
    num_rej_paper = args.num_papers - num_acc_paper
    falg = True
    while falg:
        for track in all_tracks:
            track_group = rej_group[rej_group["track"]==track]
            for idx, row in track_group.iterrows():
                if row["ID"] not in selected_rej_paper_ids:
                    selected_rej_paper_ids.append(row["ID"])
                    break
            if len(selected_rej_paper_ids) >= num_rej_paper:
                falg = False
                break
    
    print(f"==> selected {len(selected_rej_paper_ids)} rejection papers")
    
    
    selected_paper_ids = selected_acc_paper_ids + selected_rej_paper_ids
    
    with open(os.path.join(args.target_dir, args.save_file), "w") as f:
        for paper_id in selected_paper_ids:
            f.write(f"{paper_id}\n")
    
    # count the track distribution of the selected papers
    # plot a bar chart to show the track distribution of the selected papers
    # x-axis: track, y-axis: number of papers
    selected_df = final_df[final_df["ID"].isin(selected_paper_ids)]
    import matplotlib.pyplot as plt
    selected_track_distribution = selected_df["track"].value_counts()
    all_track_keys = selected_track_distribution.keys()
    # make track keys shorter (del all "(XXXX)") and display vertically
    all_track_keys = [re.sub(r"\(.*\)", "", track) for track in all_track_keys]
    key_map = {"Machine Learning": "ML", "Deep Learning": "DL"}
    all_track_keys = [re.sub(r"Machine Learning", "ML", track) for track in all_track_keys]
    all_track_keys = [re.sub(r"Deep Learning", "DL", track) for track in all_track_keys]
    plt.figure(figsize=(8, 6))
    plt.bar(all_track_keys, selected_track_distribution.values, width=0.6, color="#9abdd9", linewidth=2)  # #b3c1dc
    # make the keys font size smaller, and also make the distance between each bar larger
    # set the font as Times New Roman
    plt.xticks(rotation=275, fontsize=8, fontname="Times New Roman")
    # plt.xlabel("track")
    plt.ylabel("# Papers", fontname="Times New Roman")
    # plt.title("track distribution of the selected papers")
    # save the plot (in pdf, and tight layout)
    # Add the y values on top of each bar
    for i, v in enumerate(selected_track_distribution.values):
        plt.text(i, v + 0.5, str(v), color='black', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(args.target_dir, "track_distribution.pdf"), bbox_inches='tight')
    
    # attach an additional column to the selected_df, which is the avg score of each paper, named "overall score"
    # the overall score is calculated by the "Scores"
    # for example, if the "Scores" is [['4', '8', '4'], ['3', '8', '4'], ['3', '8', '4'], ['3', '8', '4']]
    # then the "overall score" is first taken by the second number in each list, and then calculate the average
    # in the above case, is (8+8+8+8)/4 = 8
    selected_df["overall_score"] = selected_df["scores"].apply(lambda x: sum([int(score[1]) for score in eval(x)])/len(eval(x)))
    # plot a pie chart to show the distribution of the overall score
    # first print the max and min score
    # print(f"==> max score: {selected_df['overall_score'].max()}")  # 8
    # print(f"==> min score: {selected_df['overall_score'].min()}")  # 1
    # then split range as 1. ">=1 & <3"; 2. ">=3 & <5"; 3. ">=5 & <6"; 4. ">=6 & <8"; 5. ">=8"
    labels = ["[1, 3)", "[3, 5)", "[5, 6)", "[6, 8)", "[8, 10)"]
    bins = [0.99, 2.99, 3.99, 5.99, 7.99, float('inf')]
    selected_df["score_range"] = pd.cut(selected_df["overall_score"], bins=bins, labels=labels)
    score_distribution = selected_df["score_range"].value_counts()
    # plot the pie chart
    plt.figure(figsize=(6, 6))
    # set font name as Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    # make sure the distribution is in the order of labels
    plt.pie(score_distribution[labels], autopct='%1.1f%%', startangle=140, colors=["#5975A4", "#5F9D6E", "#B55D60", "#CB8962", "#8579AA"])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Overall Score Distribution")
    # also show the labels-colors correspondence on the top right of the whole pie chart
    # also make sure the legend not overlap with the pie chart
    plt.legend(title="Score Range", loc="upper right", bbox_to_anchor=(0.6, 0.05, 0.5, 1), labels=labels, fontsize=8)
    plt.savefig(os.path.join(args.target_dir, "overall_score_distribution.pdf"), bbox_inches='tight')
    
    print("==> num of [1, 3): ", score_distribution["[1, 3)"])
    print("==> num of [3, 5): ", score_distribution["[3, 5)"])
    print("==> num of [5, 6): ", score_distribution["[5, 6)"])
    print("==> num of [6, 8): ", score_distribution["[6, 8)"])
    print("==> num of [8, inf): ", score_distribution["[8, 10)"])
    

        
        
if __name__ == "__main__":
    main()