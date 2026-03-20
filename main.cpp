#include <iostream>
#include <fstream>
#include <forward_list>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

#define interface struct

interface IDistribution { // интерфейс распределения
    virtual ~IDistribution() = default;
    virtual double density(double x) const = 0;
    virtual double randNum() const = 0;
    virtual double excess() const = 0;
    virtual double asymmetry() const = 0;
    virtual double mean() const = 0;
    virtual double dispersion() const = 0;
};

interface IPersistent { // интерфейс персистентного объекта
    virtual void save(std::ofstream& out) const = 0;
    virtual void load(std::ifstream& in) = 0;
};

class Normal : public IDistribution, public IPersistent {
    double mu_, sigma_;
    mutable std::mt19937 gen_;
public:
    Normal(double mu, double sigma, uint32_t seed = std::random_device{}())
        : mu_(mu), sigma_(sigma), gen_(seed) {
        if (sigma_ <= 0.0)
            throw std::invalid_argument("sigma must be > 0");
    }

    double density(double x) const override {
        const double pi = std::acos(-1.0);
        return (1.0 / (sigma_ * std::sqrt(2.0 * pi))) *
               std::exp(-0.5 * std::pow((x - mu_) / sigma_, 2));
    }

    double mean() const override { return mu_; }
    double dispersion() const override { return std::pow(sigma_, 2); }

    double randNum() const override {
        std::normal_distribution<double> dist(mu_, sigma_);
        return dist(gen_);
    }

    double asymmetry() const override { return 0.0; }
    double excess() const override { return 0.0; }

    void save(std::ofstream& stream) const override {
        stream << mu_ << ' ' << sigma_ << '\n';
    }

    void load(std::ifstream& stream) override {
        double mu, sigma;
        stream >> mu >> sigma;
        if (!stream)
            throw std::runtime_error("failed to load Normal");
        if (sigma <= 0.0)
            throw std::invalid_argument("sigma must be > 0");
        mu_ = mu;
        sigma_ = sigma;
    }

    void setMu(double mu) { mu_ = mu; }

    void setSigma(double sigma) {
        if (sigma <= 0.0)
            throw std::invalid_argument("sigma must be > 0");
        sigma_ = sigma;
    }
};

class Uniform : public IDistribution, public IPersistent {
    double a_, b_;
    mutable std::mt19937 gen_;
public:
    Uniform(double a, double b, uint32_t seed = std::random_device{}())
        : a_(a), b_(b), gen_(seed) {
        if (b_ <= a_)
            throw std::invalid_argument("must be b > a");
    }

    double density(double x) const override {
        if (x < a_ || x > b_) return 0.0;
        return 1.0 / (b_ - a_);
    }

    double mean() const override {
        return (a_ + b_) / 2.0;
    }

    double dispersion() const override {
        return std::pow(b_ - a_, 2) / 12.0;
    }

    double randNum() const override {
        std::uniform_real_distribution<double> dist(a_, b_);
        return dist(gen_);
    }

    double asymmetry() const override { return 0.0; }
    double excess() const override { return -6.0 / 5.0; }

    void save(std::ofstream& stream) const override {
        stream << a_ << ' ' << b_ << '\n';
    }

    void load(std::ifstream& stream) override {
        double a, b;
        stream >> a >> b;
        if (!stream)
            throw std::runtime_error("failed to load Uniform");
        if (b <= a)
            throw std::invalid_argument("must be b > a");
        a_ = a;
        b_ = b;
    }

    void setA(double a) {
        if (b_ <= a)
            throw std::invalid_argument("must be b > a");
        a_ = a;
    }

    void setB(double b) {
        if (b <= a_)
            throw std::invalid_argument("must be b > a");
        b_ = b;
    }
};

class ShiftedExponentialLaplace : public IDistribution, public IPersistent {
    double shift_, scale_, form_;
    mutable std::mt19937 gen_;

    double BaseDensity(double x) const {
        return (form_ * (form_ + 1 + std::abs(x))) /
               (2.0 * std::pow((form_ + std::abs(x)), 2.0)) *
               std::exp(-std::abs(x));
    }

    static double E1(double x) {
        if (x <= 0.0)
            throw std::invalid_argument("E1(x) is defined only for x > 0");

        // Numerical evaluation of E1(x) = integral from x to inf of exp(-t) / t dt.
        const double upper = x + 50.0;
        const int steps = 10000;
        const double h = (upper - x) / steps;

        auto f = [](double t) {
            return std::exp(-t) / t;
        };

        double sum = f(x) + f(upper);
        for (int i = 1; i < steps; ++i) {
            double t = x + i * h;
            sum += (i % 2 == 0 ? 2.0 : 4.0) * f(t);
        }

        return sum * h / 3.0;
    }

    double BaseDispersion() const {
        double e1 = E1(form_);
        double var = 1.0 - form_ * std::exp(form_) * e1;
        return 2 * form_ * var;
    }

public:
    ShiftedExponentialLaplace(double shift, double scale, double form,
                              uint32_t seed = std::random_device{}())
        : shift_(shift), scale_(scale), form_(form), gen_(seed) {
        if (scale <= 0.0)
            throw std::invalid_argument("scale must be > 0");
        if (form <= 0.0)
            throw std::invalid_argument("form must be > 0");
    }

    double density(double x) const override {
        return BaseDensity((x - shift_) / scale_) / scale_;
    }

    double randNum() const override {
        std::exponential_distribution<double> exp1(1.0);
        std::exponential_distribution<double> expForm(form_);
        std::bernoulli_distribution bern(0.5);

        double e = exp1(gen_);
        double t = expForm(gen_);
        int b = bern(gen_) ? 1 : 0;

        double x0 = e * (2 * b - 1) / (t + 1.0);
        return shift_ + scale_ * x0;
    }

    double mean() const override {
        return shift_;
    }

    double dispersion() const override {
        return scale_ * scale_ * BaseDispersion();
    }

    double asymmetry() const override {
        return 0.0;
    }

    double excess() const override {
        double e1 = E1(form_);
        double var = 1.0 - form_ * std::exp(form_) * e1;
        return (std::pow(form_, 2) * var - form_ + 2.0) /
               (form_ * std::pow(var, 2)) - 3.0;
    }

    void save(std::ofstream& stream) const override {
        stream << shift_ << ' ' << scale_ << ' ' << form_ << '\n';
    }

    void load(std::ifstream& stream) override {
        double shift, scale, form;
        stream >> shift >> scale >> form;
        if (!stream)
            throw std::runtime_error("failed to load ShiftedExponentialLaplace");
        if (scale <= 0.0)
            throw std::invalid_argument("scale must be > 0");
        if (form <= 0.0)
            throw std::invalid_argument("form must be > 0");

        shift_ = shift;
        scale_ = scale;
        form_ = form;
    }

    void setShift(double shift) { shift_ = shift; }

    void setScale(double scale) {
        if (scale <= 0.0)
            throw std::invalid_argument("scale must be > 0");
        scale_ = scale;
    }

    void setForm(double form) {
        if (form <= 0.0)
            throw std::invalid_argument("form must be > 0");
        form_ = form;
    }
};

// --------------------------------------------------------------------------

interface IObserver {
    virtual ~IObserver() = default;
    virtual void update() = 0;
};

class Data {
    std::vector<double> data;
    double min_, max_;
    std::forward_list<std::pair<IObserver*, int>> observers;

    void recomputeMinMax() {
        auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
        min_ = *min_it;
        max_ = *max_it;
    }

public:
    Data(const std::vector<double>& data) : data(data) {
        if (data.empty())
            throw std::invalid_argument("data is empty");
        recomputeMinMax();
    }

    const std::vector<double>& getData() const { return data; }
    double getMin() const { return min_; }
    double getMax() const { return max_; }
    std::size_t size() const { return data.size(); }

    double addDataPoint(double x) {
        data.push_back(x);
        if (x < min_) min_ = x;
        if (x > max_) max_ = x;
        notify(0); // автоматическое уведомление
        return x;
    }

    void changeData(int i, double x) {
        if (i < 0 || i >= static_cast<int>(data.size()))
            throw std::out_of_range("index out of range");

        double oldValue = data[i];
        data[i] = x;

        if (x < min_ || x > max_ || oldValue == min_ || oldValue == max_) {
            recomputeMinMax();
        }

        notify(0); // автоматическое уведомление
    }

    double getDataPoint(int i) const {
        if (i < 0 || i >= static_cast<int>(data.size()))
            throw std::out_of_range("index out of range");
        return data[i];
    }

    std::vector<double> countEmpiric() const {
        double mean = [this]() {
            double sum = 0.0;
            for (double x : data) sum += x;
            return sum / data.size();
        }();

        double dispersion = [this, mean]() {
            double sum = 0.0;
            for (double x : data) sum += (x - mean) * (x - mean);
            return sum / data.size();
        }();

        double asymmetry = 0.0;
        double excess = 0.0;

        if (dispersion > 0.0) {
            asymmetry = [this, mean, dispersion]() {
                double sum = 0.0;
                for (double x : data) sum += std::pow(x - mean, 3);
                return sum / (data.size() * std::pow(dispersion, 1.5));
            }();

            excess = [this, mean, dispersion]() {
                double sum = 0.0;
                for (double x : data) sum += std::pow(x - mean, 4);
                return sum / (data.size() * std::pow(dispersion, 2)) - 3.0;
            }();
        }

        return { mean, dispersion, asymmetry, excess };
    }

    void attach(IObserver* observer, int interest) {
        observers.push_front(std::make_pair(observer, interest));
    }

    void detach(IObserver* observer, int interest) {
        observers.remove(std::make_pair(observer, interest));
    }

    void notify(int interest = 1) {
        for (auto& elem : observers) {
            if (elem.second == interest && elem.first)
                elem.first->update();
        }
    }
};

class Histogram : public IObserver {
    Data& data_;
    int k_;                        // число столбцов
    std::vector<double> dens_;     // эмпирическая плотность по интервалам
    double left_;                  // левая граница
    double right_;                 // правая граница
    double h_;                     // ширина интервала
    int interest_;

public:
    Histogram(Data& d, int k, int interest = 0)
        : data_(d), k_(k), left_(0.0), right_(0.0), h_(1.0), interest_(interest) {
        if (k_ <= 0)
            throw std::invalid_argument("number of bins must be > 0");
        data_.attach(this, interest_);
        update();
    }

    ~Histogram() {
        data_.detach(this, interest_);
    }

    void update() override {
        const std::vector<double>& sample = data_.getData();
        std::size_t n = sample.size();

        if (n == 0) {
            dens_.clear();
            left_ = right_ = 0.0;
            h_ = 1.0;
            return;
        }

        left_ = data_.getMin();
        right_ = data_.getMax();

        // Если все элементы одинаковые
        if (left_ == right_) {
            left_ -= 0.5;
            right_ += 0.5;
        }

        h_ = (right_ - left_) / k_;
        dens_.assign(k_, 0.0);

        std::vector<int> counts(k_, 0);

        for (double x : sample) {
            int idx;
            if (x == right_) {
                idx = k_ - 1;
            } else {
                idx = static_cast<int>((x - left_) / h_);
                if (idx < 0) idx = 0;
                if (idx >= k_) idx = k_ - 1;
            }
            counts[idx]++;
        }

        for (int i = 0; i < k_; ++i) {
            dens_[i] = static_cast<double>(counts[i]) / (n * h_);
        }
    }

    double density(double x) const {
        if (x < left_ || x > right_) return 0.0;
        int idx;
        if (x == right_) {
            idx = k_ - 1;
        } else {
            idx = static_cast<int>((x - left_) / h_);
            if (idx < 0 || idx >= k_) return 0.0;
        }
        return dens_[idx];
    }

    const std::vector<double>& getDensities() const { return dens_; }
    int getBinsCount() const { return k_; }
    double getLeft() const { return left_; }
    double getRight() const { return right_; }
    double getStep() const { return h_; }

    void print() const {
        std::cout << "Гистограмма: k = " << k_
                  << ", интервал = [" << left_ << "; " << right_
                  << "], h = " << h_ << "\n";
        for (int i = 0; i < k_; ++i) {
            double l = left_ + i * h_;
            double r = l + h_;
            std::cout << "[" << std::setw(8) << l << "; "
                      << std::setw(8) << r << ") -> "
                      << dens_[i] << "\n";
        }
    }

};

// --------------------------------------------------------------------------
// Демонстрационные функции
// --------------------------------------------------------------------------

std::vector<double> generateSample(const IDistribution& dist, std::size_t n) {
    std::vector<double> sample;
    sample.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        sample.push_back(dist.randNum());
    }
    return sample;
}

void printCharacteristicsComparison(const std::string& title,
                                    const Data& data,
                                    const IDistribution& dist) {
    std::vector<double> emp = data.countEmpiric();

    std::cout << "\n=== " << title << " ===\n";
    std::cout << "Размер выборки: " << data.size() << "\n";
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Мат. ожидание: эмпирическое = " << emp[0]
              << ", теоретическое = " << dist.mean()
              << ", |разность| = " << std::abs(emp[0] - dist.mean()) << "\n";

    std::cout << "Дисперсия:    эмпирическая = " << emp[1]
              << ", теоретическая = " << dist.dispersion()
              << ", |разность| = " << std::abs(emp[1] - dist.dispersion()) << "\n";

    std::cout << "Асимметрия:   эмпирическая = " << emp[2]
              << ", теоретическая = " << dist.asymmetry()
              << ", |разность| = " << std::abs(emp[2] - dist.asymmetry()) << "\n";

    std::cout << "Эксцесс:      эмпирический = " << emp[3]
              << ", теоретический = " << dist.excess()
              << ", |разность| = " << std::abs(emp[3] - dist.excess()) << "\n";
}

void printDensityComparison(const std::string& title,
                            const Histogram& hist,
                            const IDistribution& dist,
                            const std::vector<double>& points) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "x\t\tf_теоретическая(x)\tf_эмпирическая(x)\n";
    for (double x : points) {
        std::cout << x << "\t\t"
                  << dist.density(x) << "\t\t"
                  << hist.density(x) << "\n";
    }
}

void observerDemo() {
    std::cout << "\n============== ДЕМОНСТРАЦИЯ НАБЛЮДАТЕЛЯ ==============\n";

    // Data data({1.0, 2.0, 3.0, 4.0, 5.0});
    Normal dist(0.0, 1.0);

    std::vector<double> sample;
    for (int i = 0; i < 100; ++i) {
        sample.push_back(dist.randNum());
    }

    Data data(sample);


    // h2 и h3 автоматически обновляются при changeData/addDataPoint (interest = 0)
    Histogram h1(data, 5, 1);
    Histogram h2(data, 10);
    Histogram h3(data, 15);

    std::cout << "\nНачальная эмпирическая плотность для x = 2.5:\n";
    std::cout << "h1: " << h1.density(2.5) << "\n";
    std::cout << "h2: " << h2.density(2.5) << "\n";
    std::cout << "h3: " << h3.density(2.5) << "\n";

    data.changeData(0, 1.2);
    data.changeData(1, 2.3);
    data.changeData(2, 3.4);

    std::cout << "\nПосле изменения трёх точек (h2 и h3 обновились автоматически):\n";
    std::cout << "h1: " << h1.density(2.5) << "\n";
    std::cout << "h2: " << h2.density(2.5) << "\n";
    std::cout << "h3: " << h3.density(2.5) << "\n";

    std::cout << "\nРучное уведомление для h1:\n";
    data.notify(1);
    std::cout << "h1: " << h1.density(2.5) << "\n";
}

void saveHistogramData(const std::string& filename,const Histogram& hist,const IDistribution& dist){
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("failed to open file for saving histogram data");
    (void)dist;

    out << "bin_left,bin_right,emp_density\n";

    const std::vector<double>& densities = hist.getDensities();
    const double left = hist.getLeft();
    const double step = hist.getStep();

    out << std::fixed << std::setprecision(6);
    for (int i = 0; i < hist.getBinsCount(); ++i) {
        double binLeft = left + i * step;
        double binRight = binLeft + step;
        out << binLeft << ',' << binRight << ',' << densities[i] << '\n';
    }
}

void demoDistribution(const std::string& name,
                      const IDistribution& dist,
                      const std::vector<std::size_t>& sizes,
                      const std::vector<double>& densityPoints,
                      int binsForHistogram = 12) {
    std::cout << "\n===============================================\n";
    std::cout << "ДЕМОНСТРАЦИЯ ДЛЯ: " << name << "\n";
    std::cout << "===============================================\n";

    for (std::size_t n : sizes) {
        std::vector<double> sample = generateSample(dist, n);
        Data data(sample);

        printCharacteristicsComparison(name + ", n = " + std::to_string(n), data, dist);

        // Сравнение плотностей делаем только для не слишком больших выборок
        if (n < 1000) {
            const int sturgesBins = std::max(1, static_cast<int>(std::ceil(1.0 + std::log2(static_cast<double>(n)))));
            Histogram hist(data, sturgesBins);
            printDensityComparison("Сравнение плотностей для " + name +
                                   ", n = " + std::to_string(n),
                                   hist, dist, densityPoints);
            saveHistogramData(name + "_histogram_data_n" + std::to_string(n) + ".csv", hist, dist);
        }
    }
}

int main() {
    try {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
#endif

        Normal normalDist(0.0, 1.0, 42);
        Uniform uniformDist(-1.0, 3.0, 42);
        ShiftedExponentialLaplace selDist(0.5, 1.2, 2.0, 42);

        // 6.1, 6.2, 6.3
        demoDistribution("Normal(0,1)",
                        normalDist,
                        {50, 500, 5000, 1000000},
                        {-2.0, -1.0, 0.0, 1.0, 2.0});

        demoDistribution("Uniform(-1,3)",
                        uniformDist,
                        {50, 500, 5000, 1000000},
                        {-1.0, 0.0, 1.0, 2.0, 3.0});

        demoDistribution("ShiftedExponentialLaplace(shift=0.5, scale=1.2, form=2.0)",
                        selDist,
                        {50, 500, 5000, 1000000},
                        {-1.0, 0.0, 0.5, 1.0, 2.0});

        // 6.4
        observerDemo();
    }
    catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
