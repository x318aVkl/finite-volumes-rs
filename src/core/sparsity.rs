


/// Represents a sparsity pattern defined by a dense set of major indices and sparse minor indices for each major entry
/// - For CSR matrices, major indices are rows, and minor indices are columns.
/// - For mesh cell-to-cell indexing, major indices are owner cells, and minor indices are neighbor cells
#[derive(Debug, Clone)]
pub struct Sparsity<T> {
    minors: Vec<T>,
    major_starts: Vec<usize>,
    max_minor: T,
}


impl<T: PartialOrd + Ord + Clone + Copy + std::ops::Add<Output = T> + From<usize>> Sparsity<T> where usize: From<T> {
    pub fn new() -> Sparsity<T> {
        Sparsity { minors: vec![], major_starts: vec![0], max_minor: 0.into(), }
    }

    pub fn with_capacity(majors: usize, minors: usize,) -> Sparsity<T> {
        let mut major_starts = Vec::with_capacity(majors);
        major_starts.push(0);
        Sparsity { minors: Vec::with_capacity(minors), major_starts: major_starts, max_minor: 0.into(), }
    }

    /// Adds a minor index to the last major index's range
    pub fn push_to_major(&mut self, minor: T) {
        self.minors.push(minor);
        self.max_minor = self.max_minor.max(minor);
    }

    /// Closes the last major index and starts an empty one
    pub fn close_major(&mut self) {
        self.major_starts.push(self.minors.len());
    }

    /// Returns the minor indices associated with a major index
    pub fn major_range<'a>(&'a self, major: usize) -> &'a [T] {
        &self.minors[self.major_starts[major]..self.major_starts[major + 1]]
    }

    /// Returns the flat indexing value into minor indices of the starting position of a major index
    pub fn major_start(&self, major: usize) -> usize {
        self.major_starts[major]
    }

    /// Returns the flat indexing value into minor indices of the ending position of a major index
    pub fn major_end(&self, major: usize) -> usize {
        self.major_starts[major + 1]
    }

    /// Returns the number of closed major indices
    pub fn major_len(&self) -> usize {
        self.major_starts.len() - 1
    }

    /// Returns the total number of minor indices
    pub fn minor_len(&self) -> usize {
        self.minors.len()
    }

    /// Returns the maximum minor index
    pub fn max_minor(&self) -> T {
        self.max_minor
    }

    /// Acces a minor index using flat indexes, like the ones returned by major_start()
    pub fn flat_index(&self, k: usize) -> T {
        self.minors[k]
    }

    /// Mutable acces a minor index using flat indexes, like the ones returned by major_start()
    pub fn flat_index_mut(&mut self, k: usize) -> &mut T {
        &mut self.minors[k]
    }

    /// Inserts a minor index to the given major index. This is O(N).
    /// - Returns: Ok(pos) if the pair (major, minor) was not already present in the sparsity. pos si a flat index representing the position of the added value in the minors data.
    /// - Returns: Err(pos) if the pair (major, minor) was already present in the sparsity. The value in minors at that position is not updated.
    pub fn insert(&mut self, major: usize, minor: T) -> Result<usize, usize> {
        let pos = match self.major_range(major).binary_search(&minor) {
            Ok(p) => {
                return Err(self.major_start(major) + p);
            },
            Err(p) => self.major_start(major) + p
        };
        self.minors.insert(pos, minor);
        for i in (major + 1)..self.major_len() {
            self.major_starts[i] += 1;
        }
        Ok(pos)
    }


    pub fn contains(&self, major: usize, minor: T) -> bool {
        for k in self.major_range(major) {
            if minor == *k {
                return true;
            }
        }
        false
    }

    pub fn find_flat_index(&self, major: usize, minor: T) -> Option<usize> {
        let start = self.major_start(major);
        let end = self.major_end(major);
        for k in start..end {
            if self.minors[k] == minor {
                return Some(k);
            }
        }
        None
    }
    pub fn find_flat_index_sorted(&self, major: usize, minor: T) -> Option<usize> {
        let start = self.major_start(major);
        let end = self.major_end(major);
        match self.minors[start..end].binary_search(&minor) {
            Ok(k) => Some(k),
            Err(_) => None,
        }
    }

    /// Sort each major range of the sparsity
    /// - Usefull to use binary seach to find minors, if order of inserting is not important. Used by CSR matrices.
    pub fn sorted(mut self) -> Self {

        self.sort();

        self
    }

    pub fn sort(&mut self) {

        for i in 0..self.major_len() {
            let rs = self.major_start(i);
            let re = self.major_end(i);
            let row = &mut self.minors[rs..re];
            row.sort();
        }
    }

    /// Sort each major range of the sparsity, along with minor-wise values.
    pub fn sorted_with<D: Clone>(mut self, mut values: Vec<D>) -> (Self, Vec<D>) {

        for i in 0..self.major_len() {
            let rs = self.major_start(i);
            let re = self.major_end(i);

            let mut merged: Vec<_> = self.minors[rs..re].iter().zip(&values[rs..re]).map(|(a, b)| {(*a, b.clone())}).collect();
            merged.sort_by(|a, b| {a.0.cmp(&b.0)});

            for k in rs..re {
                self.minors[k] = merged[k - rs].0;
                values[k] = merged[k - rs].1.clone();
            }
        }

        (self, values)
    }

}

