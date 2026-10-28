#include "gac/templates/Registry.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  llvm::outs() << llvm::formatv("{0:2}\n",
                              llvm::json::Value(gac::templates::catalog()));
}
