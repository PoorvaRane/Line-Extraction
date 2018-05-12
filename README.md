# Line-Extraction

### To perform line extraction on a page, perform the following steps in order:


##### Phase 1 (Text-box extraction)
1) To get a mask of the textual regions in a page, run:
   ```
   mkdir step1_output
   python step1.py <path_to_original_image> <approx_num_lines_in_page>
   ```


2) To perform text-box extraction from the image mask, run:
   ```
   mkdir step2_output
   python step2.py <path_to_step1_ouput_image> <path_to_original_image>
   ```
   The output of this step is two text-box regions in the directory step2_output.


##### Phase 2 (Line extraction)
3) Prepare rowsum (csv file) for input to spline fitting.
   ```
   python rowsum.py <path_to_step2_ouput_image>
   ```

4) csv file to give as input to line extractor after spline fitting.
   ```
   python line_spline.py <path_to_step3_output_csvfile>
   ```

5) To draw lines, run:
   ```
   mkdir line_output
   python draw_lines.py <path_to_original_image> <path_to_step3_output_csvfile>
   ```

6) To extract lines, run:
   ```
   python extract_lines.py <path_to_original_image> <path_to_step4_output_csvfile>
   ```

   The output of this step is individual lines in the directory line_output.