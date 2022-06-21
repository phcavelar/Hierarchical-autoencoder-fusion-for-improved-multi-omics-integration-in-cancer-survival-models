library(TCGAbiolinks)
library(here)
library(rjson)
library(tibble)
library(maftools)
library(vroom)

source(here::here("src", "prep", "prepare_tcga_data.R"))
source(here::here("src", "query", "read_tcga_data.R"))

config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)
# Make sure `vroom` has enough connection size available
# in order to read some of the largear files.
Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 2)

# Depending on your exact forge distribution, this line might break,
# if so, please adapt it to match whereever your env lives.
Sys.setenv(PATH = paste(c(paste0("/Users/", Sys.info()[["user"]], "/miniforge3/envs/hierarchical_fusion/bin"), Sys.getenv("PATH"),
  collapse = .Platform$path.sep
), collapse = ":"))

# Load all data files.
gex_master <- vroom(
  here::here(
    "data", "raw", "PANCAN", "TCGA", "gex", "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
  )
) %>% data.frame(check.names = FALSE)

cnv_master <- vroom(here::here(
  "data", "raw", "PANCAN", "TCGA", "copy_number", "TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"
)) %>% data.frame(check.names = FALSE)

meth_master <- vroom(here::here(
  "data", "raw", "PANCAN", "TCGA", "methylation", "jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv"
)) %>% data.frame(check.names = FALSE)

mirna_master <- vroom(here::here("data", "raw", "PANCAN", "TCGA", "mirna", "pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv")) %>%
  data.frame(check.names = FALSE)

rppa_master <- vroom::vroom(here::here("data", "raw", "PANCAN", "TCGA", "rppa", "TCGA-RPPA-pancan-clean.txt"))

# Calculate non silent mutation genes by patient matrix.
mutation <- maftools::read.maf(here::here("data", "raw", "PANCAN", "TCGA", "mutation", "mc3.v0.2.8.PUBLIC.maf.gz"))
mut_master <- mutCountMatrix(mutation,
  removeNonMutated = TRUE
)

clinical <- read_raw_clinical_data()
tcga_cdr <- clinical$clinical_data_resource_outcome
tcga_w_followup <- clinical$clinical_with_followup

for (cancer in config$cancers) {
  print(paste0("Starting ", cancer))
  prepare_new_cancer_dataset(cancer = cancer, include_rppa = TRUE, clinical_drop_limit = config$clinical_drop_limit, omics_drop_limit = config$omics_drop_limit, drop_first_dummy = TRUE)
  # Go to Python in order to create the splits as this is where 
  # most of the utils code lives.
  system(paste("python", "src/chores/get_new_splits.py", cancer, sep = " "))
  print(paste0("Finishing ", cancer))
}
